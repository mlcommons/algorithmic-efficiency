from typing import Tuple

from absl import app
from absl import flags
from absl import logging
import collections
import importlib
import os
import struct
import time

import jax
import spec
from workloads.mnist import workload as mnist_workload


flags.DEFINE_string(
    'submission_path',
    'workloads/mnist/submission.py',
    'The relative path of the Python file containing submission functions. '
    'NOTE: the submission dir must have an __init__.py file!')
flags.DEFINE_string('workload', 'mnist', 'The name of the workload to run.')

FLAGS = flags.FLAGS


# TODO(znado): make a nicer registry of workloads that lookup in.
WORKLOADS = {
    'mnist': mnist_workload.MnistWorkload(),
}


# Example reference implementation showing how to use the above functions
# together.
def train_once(
    workload: spec.Workload,
    init_optimizer_state: spec.InitOptimizerFn,
    update_params: spec.UpdateParamsFn,
    data_selection: spec.DataSelectionFn,
    hyperparameters: spec.Hyperparamters,
    rng: spec.RandomState) -> Tuple[spec.Timing, spec.Steps]:
  data_rng, opt_init_rng, model_init_rng, rng = jax.random.split(rng, 4)

  # Workload setup.
  input_queue = workload.build_input_queue(
      data_rng, 'train', batch_size=hyperparameters.batch_size)
  optimizer_state = init_optimizer_state(
      workload,
      workload.param_shapes,
      hyperparameters,
      opt_init_rng)
  model_params, model_state = workload.init_model_fn(
      workload.param_shapes, model_init_rng)

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
        workload.loss_type,
        hyperparameters,
        global_step,
        data_select_rng)
    (augmented_train_input_batch,
     augmented_train_label_batch) = workload.preprocess_for_train(
        selected_train_input_batch,
        selected_train_label_batch,
        workload.train_mean,
        workload.train_stddev,
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
        accumulated_submission_time > workload.max_allowed_runtime)
    # Check if submission is eligible for an untimed eval.
    if eval_now or current_time - last_eval_time >= workload.eval_period_time:
      latest_eval_result = workload.eval_model(
          model_params, model_state, eval_rng)
      last_eval_time = current_time
      eval_results.append((global_step, latest_eval_result))
      goal_reached = workload.has_reached_goal(latest_eval_result)
  return accumulated_submission_time, global_step


HyperparamtersTuple = collections.namedtuple(
    'Hyperparamters',
    ('batch_size', 'beta_1', 'beta_2', 'epsilon', 'learning_rate'))


def score_submission_on_workload(
    workload: spec.Workload,
    init_optimizer_state: spec.InitOptimizerFn,
    update_params: spec.UpdateParamsFn,
    data_selection: spec.DataSelectionFn):
  # TODO(znado): add support for tuning rulesets.
  tuning_search_space = [
      HyperparamtersTuple(1024, 0.9, 0.999, 1e-8, 1e-2),
  ]
  all_timings = []
  for hyperparameters in tuning_search_space:
    rng_seed = struct.unpack('q', os.urandom(8))[0]
    rng = jax.random.PRNGKey(rng_seed)
    # Generate a new seed from hardware sources of randomness for each trial.
    timing = train_once(
        workload,
        init_optimizer_state,
        update_params,
        data_selection,
        hyperparameters,
        rng)
    all_timings.append(timing)
  return min(all_timings, key=lambda x: x[0])


def main(_):
  # Remove the trailing '.py' and convert the filepath to a Python module.
  submission_module_path = FLAGS.submission_path[:-3].replace('/', '.')
  submission_module = importlib.import_module(submission_module_path)
  workload = WORKLOADS[FLAGS.workload]
  score = score_submission_on_workload(
      workload,
      submission_module.init_optimizer_state,
      submission_module.update_params,
      submission_module.data_selection)
  logging.info(f'{FLAGS.workload} score: {score}')


if __name__ == '__main__':
  app.run(main)
