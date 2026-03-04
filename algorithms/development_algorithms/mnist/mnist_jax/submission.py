"""Training algorithm track submission functions for MNIST."""

import functools
from typing import Any, Dict, Iterator, List, Optional, Tuple

import jax
import optax

from algoperf import jax_sharding_utils, spec


def get_batch_size(workload_name: str) -> int:
  # Return the global batch size.
  batch_sizes = {'mnist': 1024}
  return batch_sizes[workload_name]


def init_optimizer_state(
  workload: spec.Workload,
  model_params: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  rng: spec.RandomState,
) -> spec.OptimizerState:
  # Unused parameters
  del model_state
  del rng
  opt_init_fn, opt_update_fn = optax.chain(
    optax.scale_by_adam(
      b1=1.0 - hyperparameters.one_minus_beta_1,
      b2=0.999,
      eps=hyperparameters.epsilon,
    ),
    optax.scale(-hyperparameters.learning_rate),
  )
  return (opt_init_fn(model_params), opt_update_fn)


# `functools.partial` here to avoid re-compiling and hitting / thrashing jit cache on every invocation
@functools.partial(
  jax.jit,
  # First two arguments of function, not "jax-relevant" args
  static_argnums=(0, 1),
  # Args at idxs 2, 3, 4 won't be used again after invocation, and memory is recycled
  donate_argnums=(2, 3, 4),
  # How to split input args 2, 3, 4, 5, 6 across multiple devices
  # `replicate` means to duplicate it, batch_dim_sharding is doing the actual paralleization by the batch dimension
  in_shardings=(
    jax_sharding_utils.get_replicate_sharding(),  # model_state
    jax_sharding_utils.get_replicate_sharding(),  # optimizer_state
    jax_sharding_utils.get_replicate_sharding(),  # current_param_container
    jax_sharding_utils.get_batch_dim_sharding(),  # batch
    jax_sharding_utils.get_replicate_sharding(),  # rng
  ),
  # How to handle output args (replicate across all devices)
  out_shardings=(
    jax_sharding_utils.get_replicate_sharding(),  # new_optimizer_state
    jax_sharding_utils.get_replicate_sharding(),  # updated_params
    jax_sharding_utils.get_replicate_sharding(),  # new_model_state
  ),
)
def _train_step(
  workload: spec.Workload,
  opt_update_fn: optax.TransformUpdateFn,
  model_state: spec.ModelAuxiliaryState,
  optimizer_state: spec.OptimizerState,
  current_param_container: spec.ParameterContainer,
  batch: Dict[str, spec.Tensor],
  rng: spec.RandomState,
) -> Tuple[spec.OptimizerState, spec.ParameterContainer, spec.ModelAuxiliaryState]:
  def loss_fn(params: spec.ParameterContainer) -> tuple[float, spec.ModelAuxiliaryState]:
    logits_batch, new_model_state = workload.model_fn(
      params=params,
      augmented_and_preprocessed_input_batch=batch,
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      update_batch_norm=True,
      dropout_rate=0.0,  # MNIST model has no dropout and this is ignored
    )
    loss_dict = workload.loss_fn(batch['targets'], logits_batch)
    loss = loss_dict['summed'] / loss_dict['n_valid_examples']
    return loss, new_model_state

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, new_model_state), grad = grad_fn(current_param_container)
  updates, new_optimizer_state = opt_update_fn(
    grad, optimizer_state, current_param_container
  )
  updated_params = optax.apply_updates(current_param_container, updates)
  return new_optimizer_state, updated_params, new_model_state


def update_params(
  workload: spec.Workload,
  current_param_container: spec.ParameterContainer,
  current_params_types: spec.ParameterTypeTree,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  batch: Dict[str, spec.Tensor],
  loss_type: spec.LossType,
  optimizer_state: spec.OptimizerState,
  eval_results: List[Tuple[int, float]],
  global_step: int,
  rng: spec.RandomState,
  train_state: Optional[Dict[str, Any]] = None,
) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del train_state
  del eval_results
  del global_step

  optimizer_state, opt_update_fn = optimizer_state
  new_optimizer_state, updated_params, new_model_state = _train_step(
    workload,
    opt_update_fn,
    model_state,
    optimizer_state,
    current_param_container,
    batch,
    rng,
  )
  return (new_optimizer_state, opt_update_fn), updated_params, new_model_state


def prepare_for_eval(
  workload: spec.Workload,
  current_param_container: spec.ParameterContainer,
  current_params_types: spec.ParameterTypeTree,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  loss_type: spec.LossType,
  optimizer_state: spec.OptimizerState,
  eval_results: List[Tuple[int, float]],
  global_step: int,
  rng: spec.RandomState,
) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params)."""
  del workload
  del hyperparameters
  del current_params_types
  del loss_type
  del eval_results
  del global_step
  del rng
  return (optimizer_state, current_param_container, model_state)


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimizer state.
def data_selection(
  workload: spec.Workload,
  input_queue: Iterator[Dict[str, spec.Tensor]],
  optimizer_state: spec.OptimizerState,
  current_param_container: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  global_step: int,
  rng: spec.RandomState,
) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a batch of training examples and labels.
  """
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del global_step
  del rng
  return next(input_queue)
