"""Rough outline of the MLPerf Algorithmic Efficiency API."""

import time
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import spec


# Training algorithm track fixed functions.


class Workload:

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result < 0.02


# Return whether or not a key in spec.ParameterTree is the output layer parameters.
def is_output_params(param_key: ParameterKey) -> bool:
  pass


def preprocess_for_train(
    selected_raw_input_batch: spec.Tensor,
    selected_label_batch: spec.Tensor,
    train_mean: spec.Tensor,
    train_stddev: spec.Tensor,
    rng: spec.RandomState) -> spec.Tensor:
  del train_mean
  del train_stddev
  del rng
  return preprocess_for_eval(
      selected_raw_input_batch, selected_label_batch, None, None)


def preprocess_for_eval(
    raw_input_batch: spec.Tensor,
    raw_label_batch: spec.Tensor,
    train_mean: spec.Tensor,
    train_stddev: spec.Tensor) -> spec.Tensor:
  del train_mean
  del train_stddev
  return (raw_input_batch, raw_label_batch)


# InitModelFn = Callable[Tuple[spec.ParameterShapeTree, spec.Seed], spec.ParameterTree]
def init_model_fn(
    param_shapes: spec.ParameterShapeTree,
    rng: spec.RandomState) -> spec.ParameterTree:
  # return initial_params
  pass


# ModelFn = Callable[
#     Tuple[spec.ParameterTree, spec.Tensor, spec.ForwardPassMode, spec.Seed, bool], spec.Tensor]
def model_fn(
    params: spec.ParameterTree,
    augmented_and_preprocessed_input_batch: spec.Tensor,
    model_state: spec.ModelAuxillaryState,
    mode: spec.ForwardPassMode,
    rng: spec.RandomState,
    update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxillaryState]:
  # return logits_batch
  # Possible side effect of updating BN.
  pass


# LossFn = Callable[Tuple[spec.Tensor, spec.Tensor], spec.Tensor]
# Does NOT apply regularization, which is left to the submitter to do in
# `update_params`.
def loss_fn(
    label_batch: spec.Tensor,
    logits_batch: spec.Tensor,
    loss_type: spec.LossType) -> spec.Tensor:  # differentiable
  del loss_type
  one_hot_targets = jax.nn.one_hot(label_batch, 10)
  return -jnp.sum(one_hot_targets * nn.log_softmax(logits_batch), axis=-1)


# TODO(all): finish this for different rulesets
def score_submission_on_workload(workload):
  tuning_search_space = []
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
  return min(all_timings)


# Example reference implementation showing how to use the above functions
# together.
def train_once(
    workload,
    init_optimizer_state,
    update_params,
    data_selection,
    hyperparameters: Hyperparamters,
    rng: spec.RandomState) -> Tuple[Timing, Steps]:
  data_rng, opt_init_rng, model_init_rng, rng = jax.random.split(rng, 4)

  # Workload setup.
  input_queue = workload.build_input_queue(workload, data_rng)
  model_fn = workload.build_model_fn(workload)
  optimizer_state = init_optimizer_state(
      workload.params_shapes,
      hyperparameters,
      opt_init_rng)
  model_params = init_model_fn(workload.param_shapes, model_init_rng)

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
    data_select_rng, preprocess_rng, update_rng = jax.random.split(step_rng, 3)
    start_time = time.time()
    selected_train_input_batch, selected_train_label_batch = data_selection(
        input_queue,
        optimizer_state,
        model_params,
        workload.loss_type,
        hyperparameters,
        global_step,
        data_select_rng)
    augmented_train_input_batch, augmented_train_label_batch = preprocess_for_train(
        selected_train_input_batch,
        selected_train_label_batch,
        workload.train_mean,
        workload.train_stddev,
        preprocess_rng)
    try:
      optimizer_state, model_params = update_params(
          model_params,
          workload.model_params_types,
          hyperparameters,
          augmented_train_input_batch,
          augmented_train_label_batch,
          workload.loss_type,
          optimizer_state,
          eval_results,
          global_step,
          update_rng)
    except spec.TrainingCompleteError:
      eval_now = True
    global_step += 1
    current_time = time.time()
    accumulated_submission_time += current_time - start_time
    is_time_remaining = accumulated_submission_time > workload.max_allowed_runtime
    # Check if submission is eligible for an untimed eval.
    if eval_now or current_time - last_eval_time >= workload.eval_period_time:
      latest_eval_result = eval_model()
      last_eval_time = current_time
      eval_results.append((global_step, latest_eval_result))
      goal_reached = workload.has_reached_goal(
          latest_eval_result,
          workload.target_metric_value,
          workload.comparison_direction)
  return accumulated_submission_time, global_step

