"""Rough outline of the MLPerf Algorithmic Efficiency API."""

import enum
import os
import struct
import time
from typing import Any, Callable, Dict, Iterator, Tuple, Union
from operator import gt, lt

# TODO: framework-dependant imports
import jax
import jax.numpy as jnp
import numpy as np
import torch
import random
# import tensorflow as tf


class LossType(enum.Enum):
  SOFTMAX_CROSS_ENTROPY = 0
  SIGMOID_CROSS_ENTROPY = 1
  MEAN_SQUARED_ERROR = 2


class ForwardPassMode(enum.Enum):
  TRAIN = 0
  EVAL = 1
  # ... ?


class ParamType(enum.Enum):
  WEIGHT = 0
  BIAS = 1
  CONV_WEIGHT = 2
  BATCH_NORM = 3
  EMBEDDING = 4


class ComparisonDirection(enum.Enum):
  MINIMIZE = 0
  MAXIMIZE = 1


class InputQueueMode(enum.Enum):
    TRAIN = 0
    EVAL = 1

# Of course, Tensor knows its shape and dtype.
# Tensor = Union[jnp.array, np.array, tf.Tensor, ...]
Tensor = Union[jnp.array, np.array, torch.Tensor]  # DeviceArray??

# TODO(znado): variadic tuples??
Shape = Union[
    Tuple[int],
    Tuple[int, int],
    Tuple[int, int, int],
    Tuple[int, int, int, int]]
ParameterShapeTree = Dict[str, Dict[str, Shape]]

# If necessary, these can be izipped together easily given they have the same
# structure, to get an iterator over pairs of leaves.
ParameterKey = str
# Dicts can be arbitrarily nested.
ParameterTree = Dict[ParameterKey, Dict[ParameterKey, Tensor]]
ParameterTypeTree = Dict[ParameterKey, Dict[ParameterKey, ParamType]]

OutputTree = Dict[ParameterKey, Dict[ParameterKey, Tensor]]
Seed = Union[int, bytes, jax.random.PRNGKey]

OptimizerState = Any
Hyperparameters = Any
Timing = int
Steps = int

InputQueue = Iterator[Tuple[Tensor, Tensor]]
# Training algorithm track fixed functions.

def _has_reached_goal(
    eval_result: float,
    workload_target: float,
    workload_comparison_direction: ComparisonDirection) -> bool:

  op = {ComparisonDirection.MAXIMIZE: gt,
        ComparisonDirection.MINIMIZE: lt}

  return op[workload_comparison_direction](eval_result, workload_target)


# Return whether or not a key in ParameterTree is the output layer parameters.
def prng(seed: Seed) -> bytes:
  pass


def is_output_params(param_key: ParameterKey) -> bool:
  pass


def preprocess_for_train(
    selected_raw_input_batch: Tensor,
    selected_label_batch: Tensor,
    train_mean: Tensor,
    train_stddev: Tensor,
    seed: Seed) -> Tuple[Tensor, Tensor]:
  # return augmented_and_preprocessed_input_batch
  # and preprocessed_label_batch
  pass


def preprocess_for_eval(
    raw_input_batch: Tensor,
    train_mean: Tensor,
    train_stddev: Tensor) -> Tensor:
  # return preprocessed_input_batch
  pass


# InitModelFn = Callable[Tuple[ParameterShapeTree, Seed], ParameterTree]
def init_model_fn(
    param_shapes: ParameterShapeTree,
    seed: Seed) -> ParameterTree:
  # return initial_params
  pass


# ModelFn = Callable[
#     Tuple[ParameterTree, Tensor, ForwardPassMode, Seed, bool], Tensor]
def model_fn(
    params: ParameterTree,
    augmented_and_preprocessed_input_batch: Tensor,
    mode: ForwardPassMode,
    seed: Seed,
    update_batch_norm: bool) -> Tuple[OutputTree, Tensor]:
  # return logits_batch
  # Possible side effect of updating BN.
  # OutputTree needed to return intermediate activations
  pass


# Keep this separate from the loss function in order to support optimizers that
# use the logits.
def output_activation_fn(
    logits_batch: Tensor,
    loss_type: LossType) -> Tensor:
  if loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
    return jax.nn.softmax(logits_batch, axis=-1)
  if loss_type == LossType.SIGMOID_CROSS_ENTROPY:
    return jax.nn.sigmoid(logits_batch)
  if loss_type == LossType.MEAN_SQUARED_ERROR:
    return logits_batch


# LossFn = Callable[Tuple[Tensor, Tensor], Tensor]
# Does NOT apply regularization, which is left to the submitter to do in
# `update_params`.
def loss_fn(
    label_batch: Tensor,
    logits_batch: Tensor,
    loss_type: LossType) -> Tensor:  # differentiable
  # return oned_array_of_losses_per_example
  pass


# Training algorithm track submission functions.


def init_optimizer_state(
    params_shapes: ParameterShapeTree,
    hyperparameters: Hyperparameters,
    seed: Seed) -> OptimizerState:
  # return initial_optimizer_state
  pass


# Each call to this function is considered a "step".
def update_params(
    current_params: ParameterTree,
    current_params_types: ParameterTypeTree,
    hyperparameters: Hyperparameters,
    augmented_and_preprocessed_input_batch: Tensor,
    label_batch: Tensor,
    # This will define the output activation via `output_activation_fn`.
    loss_type: LossType,
    model_fn: Callable,
    optimizer_state: OptimizerState,
    global_step: int,
    seed: Seed) -> Tuple[OptimizerState, ParameterTree]:
  """Return (updated_optimizer_state, updated_params)."""
  pass


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(
    input_queue: Iterator[Tuple[Tensor, Tensor]],
    optimizer_state: OptimizerState,
    current_params: ParameterTree,
    loss_type: LossType,
    hyperparameters: Hyperparameters,
    global_step: int,
    seed: Seed) -> Tuple[Tensor, Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a single training example and label.

  We left out `current_params_types` because we do not believe that it would
  # be necessary for this function.
  """
  # return input_batch, label_batch
  # return [next(input_queue) for _ in hyperparameters['batch_size']]
  pass


# TODO(all): finish this for different rulesets
def score_submission_on_workload(workload):
  tuning_search_space = []
  all_timings = []
  for hyperparameters in tuning_search_space:
    # rng_seed = struct.unpack('q', os.urandom(8))[0]
    # rng_seed = np.sum(rng_seed)
    rng_seed = 0
    seed = jax.random.PRNGKey(rng_seed)
    # Generate a new seed from hardware sources of randomness for each trial.
    timing = train_once(
        workload,
        init_optimizer_state,
        update_params,
        data_selection,
        hyperparameters,
        seed)
    all_timings.append(timing)
  return min(all_timings)

# need batch size to initiate dataloader
def _build_input_queue(workload, batch_size, seed):
  pass


def _build_model_fn(workload):
  pass

def eval_model(
  model_fn,
  model_params: ParameterTree,
  loss_fn,
  eval_metric):
  pass


# Example reference implementation showing how to use the above functions
# together.
def train_once(
    workload,  # framework-independant
    reference,  # fixed functions reference implementation
    init_optimizer_state,
    update_params,
    data_selection,
    hyperparameters: Hyperparameters,
    seed: Seed) -> Tuple[Timing, Steps]:

  # Workload setup.
  input_queue = reference._build_input_queue(
    InputQueueMode.TRAIN,
    hyperparameters['batch_size'],
    seed)
  input_queue_eval = reference._build_input_queue(
    InputQueueMode.EVAL,
    hyperparameters['batch_size'],
    seed)
  model_fn = reference._build_model_fn()
  optimizer_state = init_optimizer_state(workload.param_shapes,
      hyperparameters, seed)
  model_params = reference.init_model_fn(workload.param_shapes, seed)

  # Bookkeeping.
  goal_reached = False
  is_time_remaining = True
  last_eval_time = time.time()
  accumulated_submission_time = 0
  eval_results = []
  global_step = 0

  step_seed_generator = random.Random(seed)

  while (is_time_remaining and not goal_reached):
    # get a random seed for each step
    step_seed = step_seed_generator.randint(0, 2**16)

    start_time = time.time()
    selected_train_input_batch, selected_train_label_batch = data_selection(
        input_queue,
        optimizer_state,
        model_params,
        workload.loss_type,
        hyperparameters,
        global_step,
        step_seed)
    augmented_train_input_batch, augmented_train_label_batch = reference.preprocess_for_train(
        selected_train_input_batch,
        selected_train_label_batch,
        workload.train_mean,
        workload.train_stddev,
        step_seed)
    optimizer_state, model_params = update_params(
        model_params,
        workload.model_params_types,
        hyperparameters,
        augmented_train_input_batch,
        augmented_train_label_batch,
        workload.loss_type,
        model_fn,
        optimizer_state,
        global_step,
        step_seed)
    global_step += 1

    current_time = time.time()
    accumulated_submission_time += current_time - start_time
    is_time_remaining = accumulated_submission_time < workload.max_allowed_runtime
    # Check if submission is eligible for an untimed eval.
    if current_time - last_eval_time >= workload.eval_period_time:
      latest_eval_result = eval_model(input_queue_eval,
                                      reference.preprocess_for_eval,
                                      model_fn,
                                      model_params,
                                      reference.eval_metric,
                                      workload.train_mean,
                                      workload.train_stddev,
                                      global_step,
                                      step_seed)
      last_eval_time = time.time()  # exclude time spent on eval in eval frequency

      # TODO: metrics with display
      print(f'Step: {global_step}, Time Elapsed {accumulated_submission_time:.2f}, '
            f'score: {latest_eval_result:.4}')

      eval_results.append(latest_eval_result)
      goal_reached = _has_reached_goal(
          latest_eval_result,
          workload.target_metric_value,
          workload.comparison_direction)
  return accumulated_submission_time, global_step


def eval_model(input_queue: InputQueue,
               preprocess_for_eval,
               model_fn,
               model_params: ParameterTree,
               eval_metric,
               train_mean: tuple,
               train_stddev: tuple,
               global_step: int,
               seed: Seed):

  metric = []
  for (input_batch, label_batch) in input_queue:
    preprocessed_input_batch = preprocess_for_eval(input_batch,
                                                   train_mean,
                                                   train_stddev)

    _, logits_batch = model_fn(model_params,
                               preprocessed_input_batch,
                               ForwardPassMode.EVAL,
                               seed,
                               False)
    metric.append(eval_metric(logits_batch, label_batch))

  # TODO: more complex metrics (e.g. mAP)
  score = np.mean(metric)
  return score

