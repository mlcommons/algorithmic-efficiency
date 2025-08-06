"""Submission file for an Schedule Free AdamW optimizer in Jax."""

import functools
from typing import Dict, Iterator, List, Tuple
import optax

from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
from optax.contrib import schedule_free_adamw
from algoperf import spec
from custom_pytorch_jax_converter import use_pytorch_weights, are_weights_equal

_GRAD_CLIP_EPS = 1e-6

HPARAMS = {
    "dropout_rate": 0.1,
    "learning_rate": 0.0025,
    "one_minus_beta1": 0.1,
    "beta2": 0.9955159689799007,
    "weight_decay": 0.08121616522670176,
    "warmup_factor": 0.02,
    "weight_lr_power": 2,
    "label_smoothing": 0.2,
    "r": 0.75,
    "eps": 1e-8,
}

def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates an AdamW optimizer and a learning rate schedule."""
  model_params
  del model_state
  del rng

  opt_init_fn, opt_update_fn = schedule_free_adamw(
      learning_rate=HPARAMS['learning_rate'],
      warmup_steps=int(HPARAMS['warmup_factor'] * workload.step_hint * 0.75),

      b1=1.0 - HPARAMS['one_minus_beta1'],
      b2=HPARAMS['beta2'],
      eps=HPARAMS['eps'],
      weight_decay=HPARAMS['weight_decay'],
      weight_lr_power=HPARAMS['weight_lr_power'],
      # state_dtype=jnp.bfloat16
  )

  model_params = jax_utils.unreplicate(model_params)
  optimizer_state = opt_init_fn(model_params)

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
  # Get correct global mean loss and grad.
  (summed_loss, n_valid_examples, grad) = lax.psum(
      (summed_loss, n_valid_examples, grad), axis_name='batch')
  loss = summed_loss / n_valid_examples
  grad = jax.tree_map(lambda x: x / n_valid_examples, grad)

  grad_norm = jnp.sqrt(
      sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

  # Extract the leaves of the pytree
  leaves = jax.tree_util.tree_leaves(grad)
  # Count the total number of elements in all leaves
  total_size = sum(jnp.size(leaf) for leaf in leaves)

  # jax.debug.print('GRAD NORM {}', grad_norm)
  # jax.debug.print('NUM PARAMS {}', total_size)

  if grad_clip is not None:
    grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
    grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
    grad = jax.tree_map(lambda x: x * grad_scaling_factor, grad)

  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
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
    
  # Log the number of parameters.
  if global_step % 100 == 0:
    date_ = "2025-07-01"
    file_name = f"/results/schedule_free_pytorch_weights/criteo1tb_{date_}_after_{global_step}_steps.pth"
    params = use_pytorch_weights(file_name=file_name)
    are_weights_equal(new_params, params)
    del params

  return (new_optimizer_state, opt_update_fn), new_params, new_model_state


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262_144
  elif workload_name == 'fastmri':
    return 32
  elif workload_name == 'imagenet_resnet':
    return 1024
  elif workload_name == 'imagenet_resnet_silu':
    return 512
  elif workload_name == 'imagenet_resnet_gelu':
    return 512
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


def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   model_state: spec.ModelAuxiliaryState,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
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

