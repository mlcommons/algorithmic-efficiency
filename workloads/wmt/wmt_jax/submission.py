"""Training algorithm track submission functions for WMT."""
import functools
import types
from typing import Iterator, List, Tuple

from flax import jax_utils
from flax import optim
from flax.training import common_utils
import jax
import numpy as np
import spec

CONFIG = types.SimpleNamespace(
    learning_rate=0.0625,
    warmup_steps=1000,
    label_smoothing=0.1,
    weight_decay=0.0,
    per_device_batch_size=16)


def get_batch_size(workload_name):
  batch_sizes = {"wmt_jax": CONFIG.per_device_batch_size}
  return batch_sizes[workload_name]


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparamters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del model_state
  del rng

  optimizer_def = optim.Adam(
      learning_rate=hyperparameters.learning_rate,
      beta1=1.0 - hyperparameters.one_minus_beta_1,
      beta2=0.98,
      eps=hyperparameters.epsilon)
  optimizer = optimizer_def.create(model_params)

  # Replicate optimizer.
  optimizer = jax_utils.replicate(optimizer)

  learning_rate_fn = workload.create_learning_rate_scheduler(
      base_learning_rate=hyperparameters.learning_rate,
      warmup_steps=CONFIG.warmup_steps)

  # compile multidevice versions of train/eval/predict step and cache init fn.
  p_train_step = jax.pmap(
      functools.partial(
          workload.train_step,
          config=workload.train_config,
          learning_rate_fn=learning_rate_fn,
          label_smoothing=CONFIG.label_smoothing),
      axis_name="batch",
      donate_argnums=(0,))  # pytype: disable=wrong-arg-types

  return optimizer, p_train_step


def update_params(
    workload: spec.Workload,
    current_param_container: spec.ParameterContainer,
    current_params_types: spec.ParameterTypeTree,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparamters,
    input_batch: spec.Tensor,
    label_batch: spec.Tensor,
    # This will define the output activation via `output_activation_fn`.
    loss_type: spec.LossType,
    optimizer_state: spec.OptimizerState,
    eval_results: List[Tuple[int, float]],
    global_step: int,
    rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params)."""
  del workload
  del current_param_container
  del current_params_types
  del eval_results
  del global_step
  del model_state
  del loss_type
  del hyperparameters
  del label_batch

  optimizer, p_train_step = optimizer_state
  dropout_rngs = jax.random.split(rng, jax.local_device_count())
  optimizer, _ = p_train_step(optimizer, input_batch, dropout_rng=dropout_rngs)

  return (optimizer, p_train_step), optimizer.target, None


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Tuple[spec.Tensor, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   hyperparameters: spec.Hyperparamters, global_step: int,
                   rng: spec.RandomState) -> Tuple[spec.Tensor, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a single training example and label.

  We left out `current_params_types` because we do not believe that it would
  # be necessary for this function.

  Return a tuple of input label batches.
  """
  del optimizer_state
  del current_param_container
  del global_step
  del rng
  del hyperparameters
  del workload

  return common_utils.shard(jax.tree_map(np.asarray, next(input_queue))), None

