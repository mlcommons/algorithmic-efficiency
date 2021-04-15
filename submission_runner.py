r"""Run a submission on a single workload.

Example command:

python3 submission_runner.py \
    --workload=mnist_jax \
    --submission_path=workloads/mnist_jax/submission.py \
    --tuning_ruleset=external \
    --tuning_search_space=workloads/mnist_jax/tuning_search_space.json \
    --num_tuning_trials=3
"""
from typing import Optional, Tuple

from absl import app
from absl import flags
from absl import logging
import importlib
import json
import os
import struct
import time

import jax
import halton
import spec
from workloads.mnist_jax import workload as mnist_jax_workload


flags.DEFINE_string(
    'submission_path',
    'workloads/mnist_jax/submission.py',
    'The relative path of the Python file containing submission functions. '
    'NOTE: the submission dir must have an __init__.py file!')
flags.DEFINE_string('workload', 'mnist', 'The name of the workload to run.')
flags.DEFINE_enum(
    'tuning_ruleset', 'external',
    enum_values=['external', 'self'],
    help='Which tuning ruleset to use.')
flags.DEFINE_string(
    'tuning_search_space',
    'workloads/mnist_jax/tuning_search_space.json',
    'The path to the JSON file describing the external tuning search space.')
flags.DEFINE_integer(
    'num_tuning_trials',
    20,
    'The number of external hyperparameter trials to run.')

FLAGS = flags.FLAGS


# TODO(znado): make a nicer registry of workloads that lookup in.
WORKLOADS = {
    'mnist_jax': mnist_jax_workload.MnistWorkload(),
}


# Example reference implementation showing how to use the above functions
# together.
def train_once(
    workload: spec.Workload,
    batch_size: int,
    init_optimizer_state: spec.InitOptimizerFn,
    update_params: spec.UpdateParamsFn,
    data_selection: spec.DataSelectionFn,
    hyperparameters: spec.Hyperparamters,
    rng: spec.RandomState) -> Tuple[spec.Timing, spec.Steps]:
  data_rng, opt_init_rng, model_init_rng, rng = jax.random.split(rng, 4)

  # Workload setup.
  input_queue = workload.build_input_queue(
      data_rng, 'train', batch_size=batch_size)
  optimizer_state = init_optimizer_state(
      workload,
      hyperparameters,
      opt_init_rng)
  model_params, model_state = workload.init_model_fn(model_init_rng)

  # Bookkeeping.
  goal_reached = False
  is_time_remaining = True
  last_eval_time = 0
  accumulated_submission_time = 0
  eval_results = []
  global_step = 0
  eval_now = False

  while (is_time_remaining and not goal_reached):
    step_rng = jax.random.fold_in(rng, global_step)
    data_select_rng, preprocess_rng, update_rng, eval_rng = jax.random.split(
        step_rng, 4)
    start_time = time.time()
    selected_train_input_batch, selected_train_label_batch = data_selection(
        workload,
        input_queue,
        optimizer_state,
        model_params,
        hyperparameters,
        global_step,
        data_select_rng)
    (augmented_train_input_batch,
     augmented_train_label_batch) = workload.preprocess_for_train(
        selected_train_input_batch,
        selected_train_label_batch,
        preprocess_rng)
    try:
      optimizer_state, model_params, model_state = update_params(
          workload=workload,
          current_params=model_params,
          current_params_types=workload.model_params_types,
          model_state=model_state,
          hyperparameters=hyperparameters,
          augmented_and_preprocessed_input_batch=augmented_train_input_batch,
          label_batch=augmented_train_label_batch,
          loss_type=workload.loss_type,
          optimizer_state=optimizer_state,
          eval_results=eval_results,
          global_step=global_step,
          rng=update_rng)
    except spec.TrainingCompleteError:
      eval_now = True
    global_step += 1
    current_time = time.time()
    accumulated_submission_time += current_time - start_time
    is_time_remaining = (
        accumulated_submission_time < workload.max_allowed_runtime_sec)
    # Check if submission is eligible for an untimed eval.
    if (current_time - last_eval_time >= workload.eval_period_time_sec or
        eval_now):
      latest_eval_result = workload.eval_model(
          model_params, model_state, eval_rng)
      last_eval_time = current_time
      eval_results.append((global_step, latest_eval_result))
      goal_reached = workload.has_reached_goal(latest_eval_result)
  metrics = {'eval_results': eval_results, 'global_step': global_step}
  return accumulated_submission_time, metrics


def score_submission_on_workload(
    workload_name: str,
    submission_path: str,
    tuning_ruleset: str,
    tuning_search_space: Optional[str] = None,
    num_tuning_trials: Optional[int] = None):
  # Remove the trailing '.py' and convert the filepath to a Python module.
  submission_module_path = FLAGS.submission_path[:-3].replace('/', '.')
  submission_module = importlib.import_module(submission_module_path)
  init_optimizer_state = submission_module.init_optimizer_state
  update_params = submission_module.update_params
  data_selection = submission_module.data_selection
  get_batch_size = submission_module.get_batch_size
  batch_size = get_batch_size(workload_name)

  workload = WORKLOADS[workload_name]

  if tuning_ruleset == 'external':
    # If the submission runner is responsible for hyperparameter tuning, load in
    # the search space and generate a list of randomly selected hyperparameter
    # settings from it.
    if tuning_search_space is None:
      raise ValueError(
          'Must provide a tuning search space JSON file when using external '
          'tuning.')
    with open(tuning_search_space, 'r') as search_space_file:
      tuning_search_space = halton.generate_search(
          json.load(search_space_file), num_tuning_trials)
    all_timings = []
    all_metrics = []
    for hyperparameters in tuning_search_space:
      # Generate a new seed from hardware sources of randomness for each trial.
      rng_seed = struct.unpack('q', os.urandom(8))[0]
      rng = jax.random.PRNGKey(rng_seed)
      timing, metrics = train_once(
          workload,
          batch_size,
          init_optimizer_state,
          update_params,
          data_selection,
          hyperparameters,
          rng)
      all_timings.append(timing)
      all_metrics.append(metrics)
    score = min(all_timings)
    for ti in range(num_tuning_trials):
      logging.info('Tuning trial %d/%d', ti + 1, num_tuning_trials)
      logging.info('Hyperparameters: %s', tuning_search_space[ti])
      logging.info('Metrics: %s', all_metrics[ti])
      logging.info('Timing: %s', all_timings[ti])
      logging.info('=' * 20)
  else:
    # If the submission is responsible for tuning itself, we only need to run it
    # once and return the total time.
    score, _ = train_once(
        workload,
        batch_size,
        init_optimizer_state,
        update_params,
        data_selection,
        hyperparameters,
        rng)
  # TODO(znado): record and return other information (number of steps).
  return score


def main(_):
  score = score_submission_on_workload(
      FLAGS.workload,
      FLAGS.submission_path,
      FLAGS.tuning_ruleset,
      FLAGS.tuning_search_space,
      FLAGS.num_tuning_trials)
  logging.info('Final %s score: %f', FLAGS.workload, score)


if __name__ == '__main__':
  app.run(main)
