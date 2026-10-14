"""Training algorithm track submission functions for MNIST."""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from algoperf import jax_sharding_utils, spec


def get_batch_size(workload_name):
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
  del model_params
  del model_state
  del rng
  params_zeros_like = jax.tree.map(
    lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes
  )
  opt_init_fn, opt_update_fn = optax.chain(
    optax.scale_by_adam(
      b1=1.0 - hyperparameters.one_minus_beta_1,
      b2=0.999,
      eps=hyperparameters.epsilon,
    ),
    optax.scale(-hyperparameters.learning_rate),
  )
  return opt_init_fn(params_zeros_like), opt_update_fn


def train_step(
  workload: spec.Workload,
  opt_update_fn,
  current_param_container: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  batch: Dict[str, spec.Tensor],
  optimizer_state: spec.OptimizerState,
  rng: spec.RandomState,
) -> spec.UpdateReturn:
  def loss_fn(params):
    logits_batch, new_model_state = workload.model_fn(
      params=params,
      augmented_and_preprocessed_input_batch=batch,
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      update_batch_norm=True,
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
  del hyperparameters

  optimizer_state, opt_update_fn = optimizer_state

  replicated = jax_sharding_utils.get_replicate_sharding()
  sharded = jax_sharding_utils.get_batch_dim_sharding()
  jitted_train_step = jax.jit(
    train_step,
    static_argnums=(0, 1),
    donate_argnums=(2, 3, 5),
    in_shardings=(
      replicated,
      replicated,
      sharded,
      replicated,
      replicated,
    ),
    out_shardings=(replicated, replicated, replicated),
  )
  new_optimizer_state, updated_params, new_model_state = jitted_train_step(
    workload,
    opt_update_fn,
    current_param_container,
    model_state,
    batch,
    optimizer_state,
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
# optimzier state.
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
