"""Submission file for a SAM optimizer with warmup+cosine LR in Jax."""

import functools
from typing import Dict, Iterator, List, Optional, Tuple

from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec

_GRAD_CLIP_EPS = 1e-6


# Copied from the official SAM GitHub repository. Note how it doesn't add an
# epsilon to the gradient norm before normalizing the gradients.
def dual_vector(y: jnp.ndarray) -> jnp.ndarray:
  """Returns the solution of max_x y^T x s.t.
  ||x||_2 <= 1.
  Args:
    y: A pytree of numpy ndarray, vector y in the equation above.
  """
  gradient_norm = jnp.sqrt(
      sum(jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)))
  normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, y)
  return normalized_gradient


# github.com/google/init2winit/blob/master/init2winit/optimizer_lib/
# sharpness_aware_minimization.py
def sharpness_aware_minimization(
    rho: float,
    grad_clip: Optional[float],
    batch_axis_name: str,
    base_opt_init_fn,
    base_opt_update_fn,
) -> optax.GradientTransformation:
  """Implementation of Sharpness Aware Minimization (SAM).
  Paper: https://arxiv.org/abs/2010.01412
  Code: https://github.com/google-research/sam
  References:
    Foret et al, 2021: https://arxiv.org/abs/2010.01412
  Args:
    rho: The size of the neighborhood for the sharpness aware minimization
      gradient updates. Defaults to 0.1.
    grad_clip: The optional value to clip the updates by. Defaults to None.
    batch_axis_name: the name of the axis to pmap over. Used to run a pmean
      before applying the optimizer update.
    base_opt_init_fn: The initialization function for the base optimizer used to
      generate updates given the total gradient.
    base_opt_update_fn: The update function for the base optimizer used to
      generate updates given the total gradient.
  Returns:
    The corresponding `GradientTransformation`.
  """

  def init_fn(params):
    return base_opt_init_fn(params)

  def update_fn(updates, state, grad_fn_params_tuple):
    (grad_fn, params) = grad_fn_params_tuple

    # Updates here have been synced (mean) across devices before being sent to
    # the optimizer. We again take the correct mean of the gradients computed on
    # the noised parameters in the same order as on the original gradients and
    # with the same 1e-6 epsilon that is used when clipping the gradients.
    updates = dual_vector(updates)
    noised_params = jax.tree_util.tree_map(
        lambda p, u: p + rho * u, params, updates)
    (_, (n_valid_examples, _)), updates = grad_fn(noised_params)
    # Get correct global mean grad.
    (n_valid_examples, updates) = lax.psum((n_valid_examples, updates),
                                           axis_name=batch_axis_name)
    updates = jax.tree_map(lambda x: x / n_valid_examples, updates)

    if grad_clip:
      updates_norm = jnp.sqrt(
          sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(updates)))
      scaled_updates = jax.tree_map(
          lambda x: x / (updates_norm + _GRAD_CLIP_EPS) * grad_clip, updates)
      updates = jax.lax.cond(updates_norm > grad_clip,
                             lambda _: scaled_updates,
                             lambda _: updates,
                             None)
    updates, state = base_opt_update_fn(updates, state, params)

    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a SAM optimizer (with AdamW base) and a learning rate schedule."""
  del model_params
  del model_state
  del rng

  def jax_cosine_warmup(step_hint: int, hyperparameters):
    # Create learning rate schedule.
    warmup_steps = int(hyperparameters.warmup_factor * step_hint)
    warmup_fn = optax.linear_schedule(
        init_value=0.,
        end_value=hyperparameters.learning_rate,
        transition_steps=warmup_steps)
    cosine_steps = max(step_hint - warmup_steps, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=hyperparameters.learning_rate, decay_steps=cosine_steps)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps])
    return schedule_fn

  # Create base optimizer + LR schedule.
  lr_schedule_fn = jax_cosine_warmup(workload.step_hint, hyperparameters)
  opt_init_fn, opt_update_fn = optax.adamw(
      learning_rate=lr_schedule_fn,
      b1=1.0 - hyperparameters.one_minus_beta1,
      b2=hyperparameters.beta2,
      eps=1e-8,
      weight_decay=hyperparameters.weight_decay)

  # Create SAM update fn.
  grad_clip = (
      hyperparameters.grad_clip
      if hasattr(hyperparameters, 'grad_clip') else None)
  opt_init_fn, opt_update_fn = sharpness_aware_minimization(
      rho=hyperparameters.rho,
      grad_clip=grad_clip,
      batch_axis_name='batch',
      base_opt_init_fn=opt_init_fn,
      base_opt_update_fn=opt_update_fn)

  # Initialize optimizer state.
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  optimizer_state = opt_init_fn(params_zeros_like)

  return jax_utils.replicate(optimizer_state), opt_update_fn


@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, None, 0, 0, 0, 0, 0, None, None),
    static_broadcasted_argnums=(0, 1),
    donate_argnums=(2, 3, 4))
def pmapped_train_step(workload,
                       opt_update_fn,
                       model_state,
                       optimizer_state,
                       current_param_container,
                       batch,
                       rng,
                       grad_clip,
                       label_smoothing):

  def _loss_fn(params, update_batch_norm=True):
    """Loss function used for training."""
    logits, new_model_state = workload.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=update_batch_norm)
    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits,
        mask_batch=batch.get('weights'),
        label_smoothing=label_smoothing)
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    return summed_loss, (n_valid_examples, new_model_state)

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  second_grad_fn = jax.value_and_grad(
      functools.partial(_loss_fn, update_batch_norm=False), has_aux=True)
  (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(
      current_param_container)
  # Get correct global mean loss and grad.
  (summed_loss, n_valid_examples, grad) = lax.psum(
      (summed_loss, n_valid_examples, grad), axis_name='batch')
  loss = summed_loss / n_valid_examples
  grad = jax.tree_map(lambda x: x / n_valid_examples, grad)

  grad_norm = jnp.sqrt(
      sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

  updates, new_optimizer_state = opt_update_fn(
      grad, optimizer_state, (second_grad_fn, current_param_container))
  updated_params = optax.apply_updates(current_param_container, updates)
  return new_optimizer_state, updated_params, new_model_state, loss, grad_norm


def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
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
  outputs = pmapped_train_step(workload,
                               opt_update_fn,
                               model_state,
                               optimizer_state,
                               current_param_container,
                               batch,
                               per_device_rngs,
                               grad_clip,
                               label_smoothing)
  new_optimizer_state, new_params, new_model_state, loss, grad_norm = outputs

  # Log loss, grad_norm.
  if global_step % 100 == 0 and workload.metrics_logger is not None:
    workload.metrics_logger.append_scalar_metrics(
        {
            'loss': loss[0],
            'grad_norm': grad_norm[0],
        }, global_step)
  return (new_optimizer_state, opt_update_fn), new_params, new_model_state


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262_144
  elif workoad_name == 'criteo1tb_layernorm':
    return 262_144
  elif workload_name == 'criteo1tb_resnet':
    return 262_144
  elif workload_name == 'fastmri':
    return 32
  elif workload_name == 'imagenet_resnet':
    return 1024
  elif workload_name == 'imagenet_vit':
    return 1024
  elif workload_name == 'librispeech_conformer':
    return 256
  elif workload_name == 'librispeech_deepspeech':
    return 256
  elif workload_name == 'ogbg':
    return 512
  elif workload_name == 'wmt':
    return 128
  elif workload_name == 'mnist':
    return 16
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')


def data_selection(
    workload: spec.Workload,
    input_queue: Iterator[Dict[str, spec.Tensor]],
    optimizer_state: spec.OptimizerState,
    current_param_container: spec.ParameterContainer,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparameters,
    global_step: int,
    rng: spec.RandomState) -> Tuple[spec.Tensor, spec.Tensor, spec.Tensor]:
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
  batch = next(input_queue)
  return batch
