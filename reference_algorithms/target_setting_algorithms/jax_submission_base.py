"""Update submission function in Jax."""
import functools
from typing import Any, Dict, List, Optional, Tuple

import jax
from jax import lax
import jax.numpy as jnp
import optax

from algoperf import spec, sharding_utils

_GRAD_CLIP_EPS = 1e-6


def train_step(workload,
               opt_update_fn,
               model_state,
               optimizer_state,
               current_param_container,
               batch,
               rng,
               grad_clip,
               label_smoothing):

  def _loss_fn(params):
    """Loss function used for training."""
    logits, new_model_state = workload.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True)
    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits,
        mask_batch=batch.get('weights'),
        label_smoothing=label_smoothing)
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    return summed_loss, (n_valid_examples, new_model_state)

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(
      current_param_container)
  # Compute mean loss and grad
  loss = summed_loss / n_valid_examples
  grad = jax.tree.map(lambda x: x / n_valid_examples, grad)

  grad_norm = jnp.sqrt(
      sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

  if grad_clip is not None:
    grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
    grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
    grad = jax.tree.map(lambda x: x * grad_scaling_factor, grad)

  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
  updated_params = optax.apply_updates(current_param_container, updates)
  return new_optimizer_state, updated_params, new_model_state, loss, grad_norm


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
    train_state: Optional[Dict[str, Any]] = None) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del train_state
  del eval_results

  optimizer_state, opt_update_fn = optimizer_state
  per_device_rngs = jax.random.split(rng, jax.local_device_count())
  if hasattr(hyperparameters, 'label_smoothing'):
    label_smoothing = hyperparameters.label_smoothing
  else:
    label_smoothing = 0.0
  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
  else:
    grad_clip = None
  mesh = sharding_utils.get_mesh()
  # Create shardings for each argument
  replicated = sharding_utils.get_replicated_sharding(mesh)  # No partitioning
  sharded = sharding_utils.get_naive_sharding_spec(mesh)  # Partition along batch dimension

  # Create the sharding rules for each argument
  arg_shardings = (
      # workload is static
      # opt_update_fn is static
      replicated,  # model_state
      replicated,  # optimizer_state
      replicated,  # current_param_container
      sharded,  # batch
      replicated,  # rng
      replicated,  # grad_clip
      replicated  # label_smoothing
  )
  out_shardings = (
      replicated,  # new_optimizer_state
      replicated,  # updated_params
      replicated,  # new_model_state
      replicated,  # loss
      replicated  # grad_norm
  )
  
  # Jit with shardings
  jitted_train_step = jax.jit(
      train_step,
      static_argnums=(0, 1),
      donate_argnums=(2, 3, 4),
      in_shardings=arg_shardings,
      out_shardings=out_shardings)
  
  new_optimizer_state, new_params, new_model_state, loss, grad_norm = jitted_train_step(
      workload, opt_update_fn, model_state, optimizer_state,
      current_param_container, batch, rng, grad_clip,
      label_smoothing)

  # Log loss, grad_norm.
  if ((global_step <= 100 or global_step % 500 == 0) and
      workload.metrics_logger is not None):
    workload.metrics_logger.append_scalar_metrics(
        {
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
        }, global_step)
  return (new_optimizer_state, opt_update_fn), new_params, new_model_state


def prepare_for_eval(workload: spec.Workload,
                     current_param_container: spec.ParameterContainer,
                     current_params_types: spec.ParameterTypeTree,
                     model_state: spec.ModelAuxiliaryState,
                     hyperparameters: spec.Hyperparameters,
                     loss_type: spec.LossType,
                     optimizer_state: spec.OptimizerState,
                     eval_results: List[Tuple[int, float]],
                     global_step: int,
                     rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params)."""
  del workload
  del hyperparameters
  del current_params_types
  del loss_type
  del eval_results
  del global_step
  del rng
  return (optimizer_state, current_param_container, model_state)
