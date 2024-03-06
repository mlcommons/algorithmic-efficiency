"""MLPerf™ Algorithmic Efficiency API."""

import abc
import enum
import functools
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from absl import logging
import jax
from torch import nn
import torch.nn.functional as F


class LossType(enum.Enum):
  SOFTMAX_CROSS_ENTROPY = 0
  SIGMOID_CROSS_ENTROPY = 1
  MEAN_SQUARED_ERROR = 2
  CTC_LOSS = 3
  MEAN_ABSOLUTE_ERROR = 4


class ForwardPassMode(enum.Enum):
  TRAIN = 0
  EVAL = 1
  # ... ?


class ParameterType(enum.Enum):
  WEIGHT = 0
  BIAS = 1
  CONV_WEIGHT = 2
  BATCH_NORM_SCALE = 3
  BATCH_NORM_BIAS = 4
  LAYER_NORM_SCALE = 5
  LAYER_NORM_BIAS = 6
  EMBEDDING = 7
  ATTENTION_Q = 8
  ATTENTION_K = 9
  ATTENTION_V = 10
  ATTENTION_OUT = 11
  ATTENTION_QKV = 12  # This is used for implementations that fuse QKV together.
  ATTENTION_KV = 13  # This is used for implementations that fuse KV together.
  # We sometimes need to split this out because otherwise fused models will have
  # a different number of biases.
  ATTENTION_BIAS = 14


# Of course, Tensor knows its shape and dtype.
# Tensor = Union[jnp.array, np.array, tf.Tensor, torch.Tensor, ...]
Tensor = Any


# Define this so that if using pytree iteration utilities, can iterate over the
# model shapes pytree without iterating over the shape tuples.
class ShapeTuple:

  def __init__(self, shape_tuple):
    self.shape_tuple = shape_tuple

  def __repr__(self):
    return f'ShapeTuple({self.shape_tuple})'

  def __eq__(self, other):
    return self.shape_tuple == other.shape_tuple


Shape = Union[Tuple[int],
              Tuple[int, int],
              Tuple[int, int, int],
              Tuple[int, int, int, int],
              ShapeTuple]
ParameterShapeTree = Dict[str, Dict[str, Shape]]

# If necessary, these can be zipped together easily given they have the same
# structure, to get an iterator over pairs of leaves.
ParameterKey = str
# Dicts can be arbitrarily nested.
ParameterContainer = Union[Dict[ParameterKey, Dict[ParameterKey, Tensor]],
                           nn.Module]
ParameterTypeTree = Dict[ParameterKey, Dict[ParameterKey, ParameterType]]

RandomState = Any  # Union[jax.random.PRNGKey, int, bytes, ...]

OptimizerState = Union[Dict[str, Any], Tuple[Any, Any]]
Hyperparameters = Any
Timing = int
Steps = int

# BN EMAs.
ModelAuxiliaryState = Any
ModelInitState = Tuple[ParameterContainer, ModelAuxiliaryState]


class Workload(metaclass=abc.ABCMeta):

  def __init__(self, *args, **kwargs) -> None:
    del args
    del kwargs
    self._param_shapes: Optional[ParameterShapeTree] = None
    self._param_types: Optional[ParameterTypeTree] = None
    self._eval_iters: Dict[str, Iterator] = {}
    self.metrics_logger = None
    # Is set in submission_runner.py for workloads with PyTorch data loaders.
    self._num_workers = None

  @property
  @abc.abstractmethod
  def target_metric_name(self) -> str:
    """The name of the target metric (useful for scoring/processing code)."""

  @abc.abstractmethod
  def has_reached_validation_target(self, eval_result: Dict[str,
                                                            float]) -> bool:
    """Return whether or not the workload validation goal has been reached."""

  @abc.abstractmethod
  def has_reached_test_target(self, eval_result: Dict[str, float]) -> bool:
    """Return whether or not the workload test goal has been reached."""

  @abc.abstractmethod
  def _build_input_queue(
      self,
      data_rng: RandomState,
      split: str,
      data_dir: str,
      global_batch_size: int,
      cache: Optional[bool] = None,
      repeat_final_dataset: Optional[bool] = None,
      num_batches: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Build the input queue for the workload data.

    This is the only function that is NOT allowed to be called by submitters.

    For Jax this should return an itertor over tensors of shape
    (num_devices, per_device_batch_size, ...), and for PyTorch this should
    return tensors of shape (global_batch_size, ...).

    The required keys are 'inputs' and 'targets', and in general the naming
    convention should be plural key names because the values are batches of
    examples.
    """

  def attach_metrics_logger(self, metrics_logger) -> None:
    """Attaches a metric logger to workload."""
    self.metrics_logger = metrics_logger
    return

  @property
  @abc.abstractmethod
  def validation_target_value(self) -> float:
    """The validation target value to reach."""

  @property
  @abc.abstractmethod
  def test_target_value(self) -> float:
    """The test target value to reach."""

  @property
  @abc.abstractmethod
  def loss_type(self) -> LossType:
    """The type of loss function."""

  @property
  @abc.abstractmethod
  def num_train_examples(self) -> int:
    """The size of the training set."""

  @property
  @abc.abstractmethod
  def eval_batch_size(self) -> int:
    """The batch size for evaluation."""

  @property
  @abc.abstractmethod
  def num_eval_train_examples(self) -> int:
    """The number of training examples to evaluate metrics on."""

  @property
  @abc.abstractmethod
  def num_validation_examples(self) -> int:
    """The size of the validation set."""

  @property
  @abc.abstractmethod
  def num_test_examples(self) -> int:
    """The size of the test set."""

  @property
  @abc.abstractmethod
  def train_mean(self) -> Any:
    """The mean of the training data."""

  @property
  @abc.abstractmethod
  def train_stddev(self) -> Any:
    """The stddev of the training data."""

  @property
  @abc.abstractmethod
  def max_allowed_runtime_sec(self) -> int:
    """The max allowed runtime of the workload in seconds."""

  @property
  @abc.abstractmethod
  def eval_period_time_sec(self) -> int:
    """The eval period of the workload in seconds."""

  @property
  @abc.abstractmethod
  def step_hint(self) -> int:
    """Max num steps the baseline algo was given to reach the target."""

  @property
  def param_shapes(self):
    """The shapes of the parameters in the workload model."""
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    return self._param_shapes

  @property
  def model_params_types(self):
    """The types of the parameters in the workload model."""
    if self._param_types is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_types!')
    return self._param_types

  @abc.abstractmethod
  def is_output_params(self, param_key: ParameterKey) -> bool:
    """Whether a key in ParameterContainer is the output layer parameters."""

  # InitModelFn = Callable[
  #     Tuple[RandomState, Optional[float], Optional[float]],
  #     ParameterContainer]
  @abc.abstractmethod
  def init_model_fn(self,
                    rng: RandomState,
                    dropout_rate: Optional[float] = None,
                    aux_dropout_rate: Optional[float] = None) -> ModelInitState:
    """Return (initial_params, initial_model_state)."""

  # ModelFn = Callable[
  #     Tuple[
  #         ParameterContainer,
  #         Dict[str, Tensor],
  #         ModelAuxiliaryState,
  #         ForwardPassMode,
  #         RandomState,
  #         bool],
  #     Tensor]
  @abc.abstractmethod
  def model_fn(self,
               params: ParameterContainer,
               augmented_and_preprocessed_input_batch: Dict[str, Tensor],
               model_state: ModelAuxiliaryState,
               mode: ForwardPassMode,
               rng: RandomState,
               update_batch_norm: bool) -> Tuple[Tensor, ModelAuxiliaryState]:
    """Return logits_batch"""
    # Possible side effect of updating BN.

  def output_activation_fn(self, logits_batch: Tensor,
                           framework: str) -> Tensor:
    """Turn logits into probabilities, according to the loss_type property."""
    if framework not in ['pytorch', 'jax']:
      raise ValueError(
          f'`framework` has to be either `pytorch` or `jax`, got {framework}.')
    activation_fn = {
        LossType.MEAN_SQUARED_ERROR: lambda z: z,
        LossType.MEAN_ABSOLUTE_ERROR: lambda z: z,
    }
    is_pytorch = framework == 'pytorch'  # If False, framework == 'jax'.
    softmax_fn = (
        functools.partial(F.softmax, dim=-1) if is_pytorch else jax.nn.softmax)
    sigmoid_fn = F.sigmoid if is_pytorch else jax.nn.sigmoid
    activation_fn[LossType.SOFTMAX_CROSS_ENTROPY] = softmax_fn
    activation_fn[LossType.SIGMOID_CROSS_ENTROPY] = sigmoid_fn
    activation_fn[LossType.CTC_LOSS] = softmax_fn
    return activation_fn[self.loss_type](logits_batch)

  # LossFn = Callable[Tuple[Tensor, Tensor], Tensor]
  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  @abc.abstractmethod
  def loss_fn(
      self,
      # Dense or one-hot labels, or a tuple of (tensor, padding) for speech.
      label_batch: Union[Tuple[Tensor, Tensor], Tensor],
      logits_batch: Union[Tuple[Tensor, Tensor], Tensor],
      mask_batch: Optional[Tensor] = None,
      label_smoothing: float = 0.0) -> Dict[str, Tensor]:  # differentiable
    """Evaluate the (masked) loss function at (label_batch, logits_batch).

    Return {'summed': scalar summed loss, 'n_valid_examples': scalar number of
    valid examples in batch, 'per_example': 1-d array of per-example losses}
    (not synced across devices).
    """

  @abc.abstractmethod
  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: ParameterContainer,
                           model_state: ModelAuxiliaryState,
                           rng: RandomState,
                           data_dir: str,
                           global_step: int = 0) -> Dict[str, float]:
    """Evaluate the model on a given dataset split, return final scalars."""

  def eval_model(self,
                 global_batch_size: int,
                 params: ParameterContainer,
                 model_state: ModelAuxiliaryState,
                 rng: RandomState,
                 data_dir: str,
                 imagenet_v2_data_dir: Optional[str],
                 global_step: int) -> Dict[str, float]:
    """Run a full evaluation of the model."""
    logging.info('Evaluating on the training split.')
    train_metrics = self._eval_model_on_split(
        split='eval_train',
        num_examples=self.num_eval_train_examples,
        global_batch_size=global_batch_size,
        params=params,
        model_state=model_state,
        rng=rng,
        data_dir=data_dir,
        global_step=global_step)
    eval_metrics = {'train/' + k: v for k, v in train_metrics.items()}
    # We always require a validation set.
    logging.info('Evaluating on the validation split.')
    validation_metrics = self._eval_model_on_split(
        'validation',
        num_examples=self.num_validation_examples,
        global_batch_size=global_batch_size,
        params=params,
        model_state=model_state,
        rng=rng,
        data_dir=data_dir,
        global_step=global_step)
    for k, v in validation_metrics.items():
      eval_metrics['validation/' + k] = v
    eval_metrics['validation/num_examples'] = self.num_validation_examples
    # Evaluate on the test set. TODO(znado): always eval on the test set.
    try:
      if self.num_test_examples is not None:
        logging.info('Evaluating on the test split.')
        test_metrics = self._eval_model_on_split(
            'test',
            num_examples=self.num_test_examples,
            global_batch_size=global_batch_size,
            params=params,
            model_state=model_state,
            rng=rng,
            data_dir=imagenet_v2_data_dir if imagenet_v2_data_dir else data_dir,
            global_step=global_step)
        for k, v in test_metrics.items():
          eval_metrics['test/' + k] = v
        eval_metrics['test/num_examples'] = self.num_test_examples
    except NotImplementedError:
      pass

    return eval_metrics


class TrainingCompleteError(Exception):
  pass


# Training algorithm track submission functions, to be filled in by the
# submitter.

InitOptimizerFn = Callable[[
    Workload,
    ParameterContainer,
    ModelAuxiliaryState,
    Hyperparameters,
    RandomState
],
                           OptimizerState]


def init_optimizer_state(workload: Workload,
                         model_params: ParameterContainer,
                         model_state: ModelAuxiliaryState,
                         hyperparameters: Hyperparameters,
                         rng: RandomState) -> OptimizerState:
  # return initial_optimizer_state
  pass


UpdateReturn = Tuple[OptimizerState, ParameterContainer, ModelAuxiliaryState]
UpdateParamsFn = Callable[[
    Workload,
    ParameterContainer,
    ParameterTypeTree,
    ModelAuxiliaryState,
    Hyperparameters,
    Dict[str, Tensor],
    LossType,
    OptimizerState,
    List[Tuple[int, float]],
    int,
    RandomState
],
                          UpdateReturn]


# Each call to this function is considered a "step".
# Can raise a TrainingCompleteError if it believes it has achieved the goal and
# wants to end the run and receive a final free eval. It will not be restarted,
# and if has not actually achieved the goal then it will be considered as not
# achieved the goal and get an infinite time score. Most submissions will likely
# wait until the next free eval and not use this functionality.
def update_params(workload: Workload,
                  current_param_container: ParameterContainer,
                  current_params_types: ParameterTypeTree,
                  model_state: ModelAuxiliaryState,
                  hyperparameters: Hyperparameters,
                  batch: Dict[str, Tensor],
                  loss_type: LossType,
                  optimizer_state: OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: RandomState) -> UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  pass


DataSelectionFn = Callable[[
    Workload,
    Iterator[Dict[str, Any]],
    OptimizerState,
    ParameterContainer,
    LossType,
    Hyperparameters,
    int,
    RandomState
],
                           Tuple[Tensor, Tensor]]


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(workload: Workload,
                   input_queue: Iterator[Dict[str, Any]],
                   optimizer_state: OptimizerState,
                   current_param_container: ParameterContainer,
                   model_state: ModelAuxiliaryState,
                   hyperparameters: Hyperparameters,
                   global_step: int,
                   rng: RandomState) -> Dict[str, Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a batch of training examples and labels.
  """
  # return next(input_queue)
  pass


def get_batch_size(workload_name: str) -> int:
  """Return the global batch size to use for a given workload."""
  pass
