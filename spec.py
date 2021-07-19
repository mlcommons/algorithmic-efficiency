"""MLPerf™ Algorithmic Efficiency API."""

import enum
import time
from typing import Any, Callable, Dict, Iterator, List, Tuple, Union

import abc
import jax


class LossType(enum.Enum):
  SOFTMAX_CROSS_ENTROPY = 0
  SIGMOID_CROSS_ENTROPY = 1
  MEAN_SQUARED_ERROR = 2


class ForwardPassMode(enum.Enum):
  TRAIN = 0
  EVAL = 1
  # ... ?


class ParameterType(enum.Enum):
  WEIGHT = 0
  BIAS = 1
  CONV_WEIGHT = 2
  BATCH_NORM = 3
  EMBEDDING = 4


# Of course, Tensor knows its shape and dtype.
# Tensor = Union[jnp.array, np.array, tf.Tensor, ...]
Tensor = Any


# Define this so that if using pytree iteration utilities, can iterate over the
# model shapes pytree without iterating over the shape tuples.
class ShapeTuple:

  def __init__(self, shape_tuple):
    self.shape_tuple = shape_tuple

Shape = Union[
    Tuple[int],
    Tuple[int, int],
    Tuple[int, int, int],
    Tuple[int, int, int, int],
    ShapeTuple]
ParameterShapeTree = Dict[str, Dict[str, Shape]]

# If necessary, these can be izipped together easily given they have the same
# structure, to get an iterator over pairs of leaves.
ParameterKey = str
# Dicts can be arbitrarily nested.
ParameterContainer = Dict[ParameterKey, Dict[ParameterKey, Tensor]]
ParameterTypeTree = Dict[ParameterKey, Dict[ParameterKey, ParameterType]]

RandomState = Any  # Union[jax.random.PRNGKey, int, bytes, ...]

OptimizerState = Any
Hyperparamters = Any
Timing = int
Steps = int

# BN EMAs.
ModelAuxillaryState = Any
ModelInitState = Tuple[ParameterContainer, ModelAuxillaryState]


UpdateReturn = Tuple[
    OptimizerState, ParameterContainer, ModelAuxillaryState]
InitOptimizerFn = Callable[
    [ParameterShapeTree, Hyperparamters, RandomState],
    OptimizerState]
UpdateParamsFn = Callable[
    [
        ParameterContainer,
        ParameterTypeTree,
        ModelAuxillaryState,
        Hyperparamters,
        Tensor,
        Tensor,
        LossType,
        OptimizerState,
        List[Tuple[int, float]],
        int,
        RandomState
    ],
    UpdateReturn]
DataSelectionFn = Callable[
    [
        Iterator[Tuple[Tensor, Tensor]],
        OptimizerState,
        ParameterContainer,
        LossType,
        Hyperparamters,
        int,
        RandomState
    ],
    Tuple[Tensor, Tensor]]


class Workload(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def has_reached_goal(self, eval_result: float) -> bool:
    """Return whether or not the workload goal has been reached."""

  @abc.abstractmethod
  def build_input_queue(
      self,
      data_rng: RandomState,
      split: str,
      data_dir: str,
      batch_size: int):
    """Build the input queue for the workload data.

    This is the only function that is NOT allowed to be called by submitters.
    """

  @abc.abstractmethod
  def param_shapes(self):
    """The shapes of the parameters in the workload model."""

  @abc.abstractmethod
  def model_params_types(self) -> ParameterType:
    """The types of the parameters in the workload model."""

  @abc.abstractproperty
  def loss_type(self):
    """The type of loss function."""

  @abc.abstractproperty
  def train_mean(self):
    """The mean of the training data."""

  @abc.abstractproperty
  def train_stddev(self):
    """The stddev of the training data."""

  @abc.abstractproperty
  def max_allowed_runtime_sec(self):
    """The max allowed runtime of the workload in seconds."""

  @abc.abstractproperty
  def eval_period_time_sec(self):
    """The eval period of the workload in seconds."""

  @abc.abstractmethod
  def is_output_params(self, param_key: ParameterKey) -> bool:
    """Whether or not a key in ParameterContainer is the output layer parameters."""

  # InitModelFn = Callable[
  #     Tuple[ParameterShapeTree, RandomState], ParameterContainer]
  @abc.abstractmethod
  def init_model_fn(
      self, rng: RandomState) -> Tuple[ParameterContainer, ModelAuxillaryState]:
    """return initial_params, initial_model_state"""

  # ModelFn = Callable[
  #     Tuple[ParameterContainer, Tensor, ForwardPassMode, RandomState, bool],
  #     Tensor]
  @abc.abstractmethod
  def model_fn(
      self,
      params: ParameterContainer,
      input_batch: Tensor,
      model_state: ModelAuxillaryState,
      mode: ForwardPassMode,
      rng: RandomState,
      update_batch_norm: bool) -> Tuple[Tensor, ModelAuxillaryState]:
    """return logits_batch"""
    # Possible side effect of updating BN.

  # Keep this separate from the loss function in order to support optimizers
  # that use the logits.
  def output_activation_fn(
      self,
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
  @abc.abstractmethod
  def loss_fn(
      self,
      label_batch: Tensor,  # Dense (not one-hot) labels.
      logits_batch: Tensor) -> Tensor:  # differentiable
    """return oned_array_of_losses_per_example"""

  @abc.abstractmethod
  def eval_model(
      self,
      params: ParameterContainer,
      model_state: ModelAuxillaryState,
      rng: RandomState):
    """Run a full evaluation of the model."""


class TrainingCompleteError(Exception):
  pass


# Training algorithm track submission functions, to be filled in by the
# submitter.


def init_optimizer_state(
    workload: Workload,
    model_params: ParameterContainer,
    model_state: ModelAuxillaryState,
    hyperparameters: Hyperparamters,
    rng: RandomState) -> OptimizerState:
  # return initial_optimizer_state
  pass


_UpdateReturn = Tuple[
    OptimizerState, ParameterContainer, ModelAuxillaryState]
# Each call to this function is considered a "step".
# Can raise a TrainingCompleteError if it believe it has achieved the goal and
# wants to end the run and receive a final free eval. It will not be restarted,
# and if has not actually achieved the goal then it will be considered as not
# achieved the goal and get an infinite time score. Most submissions will likely
# wait until the next free eval and not use this functionality.
def update_params(
    workload: Workload,
    current_params: ParameterContainer,
    current_params_types: ParameterTypeTree,
    model_state: ModelAuxillaryState,
    hyperparameters: Hyperparamters,
    input_batch: Tensor,
    label_batch: Tensor,  # Dense (not one-hot) labels.
    # This will define the output activation via `output_activation_fn`.
    loss_type: LossType,
    optimizer_state: OptimizerState,
    eval_results: List[Tuple[int, float]],
    global_step: int,
    rng: RandomState) -> _UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  pass


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(
    workload: Workload,
    input_queue: Iterator[Tuple[Tensor, Tensor]],
    optimizer_state: OptimizerState,
    current_params: ParameterContainer,
    hyperparameters: Hyperparamters,
    global_step: int,
    rng: RandomState) -> Tuple[Tensor, Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a single training example and label.

  We left out `current_params_types` because we do not believe that it would
  # be necessary for this function.
  """
  # return input_batch, label_batch (dense (not one-hot) labels)
  pass


def get_batch_size(workload_name):
  """Return a batch size to use for a given workload."""
  pass
