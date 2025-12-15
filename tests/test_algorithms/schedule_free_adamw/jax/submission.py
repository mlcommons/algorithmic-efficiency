"""Submission file for a Schedule Free AdamW optimizer in Jax."""

from typing import Dict, Iterator, List, Tuple
from functools import partial

from flax import jax_utils
import jax
import jax.numpy as jnp
import optax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from algoperf import spec
# Ensure this import matches your file structure
from .schedule_free_optax import schedule_free_adamw 

_GRAD_CLIP_EPS = 1e-6

HPARAMS = {
    'dropout_rate': 0.1,
    'learning_rate': 0.0025,
    'one_minus_beta1': 0.1,
    'beta2': 0.9955159689799007,
    'weight_decay': 0.08121616522670176,
    'warmup_factor': 0.02,
    'weight_lr_power': 2,
    'label_smoothing': 0.2,
    'r': 0.75,
    'eps': 1e-8,
}

def init_optimizer_state(
    workload: spec.Workload,
    model_params: spec.ParameterContainer,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparameters,
    rng: spec.RandomState,
) -> spec.OptimizerState:
  """Creates Schedule Free AdamW optimizer and state."""
  del model_state
  del rng
  del hyperparameters

  opt_init_fn, opt_update_fn = schedule_free_adamw(
      learning_rate=HPARAMS['learning_rate'],
      warmup_steps=int(HPARAMS['warmup_factor'] * workload.step_hint * 0.75),
      b1=1.0 - HPARAMS['one_minus_beta1'],
      b2=HPARAMS['beta2'],
      eps=HPARAMS['eps'],
      weight_decay=HPARAMS['weight_decay'],
      weight_lr_power=HPARAMS['weight_lr_power'],
  )

  optimizer_state = opt_init_fn(model_params)
  return optimizer_state, opt_update_fn


def train_step(
    workload,
    opt_update_fn,
    model_state,
    optimizer_state,
    current_param_container,
    batch,
    rng,
    grad_clip,
    label_smoothing,
    beta1
):
  
  # 1. y = (1-beta1)z + beta1*x
  z = optimizer_state.z
  def interpolate(x, z):
    z = z.astype(x.dtype) 
    return beta1 * x + (1.0 - beta1) * z

  params_y = jax.tree_util.tree_map(interpolate, current_param_container, z)

  # 2. Loss Function
  def _loss_fn(params):
    logits, new_model_state = workload.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True,
    )
    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits,
        mask_batch=batch.get('weights'),
        label_smoothing=label_smoothing,
    )
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    return summed_loss, (n_valid_examples, new_model_state)

  # 3. Gradients
  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(params_y)

  # 4. Sync
  loss = summed_loss / n_valid_examples

  grad = jax.tree_util.tree_map(lambda x: x / n_valid_examples, grad)

  # 5. Clip
  grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

  if grad_clip is not None:
    grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
    grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
    grad = jax.tree_util.tree_map(lambda x: x * grad_scaling_factor, grad)

  # 6. Update
  updates, new_optimizer_state = opt_update_fn(
      grad, optimizer_state, current_param_container
  )
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
    **kwargs,
) -> spec.UpdateReturn:
  
  optimizer_state, opt_update_fn = optimizer_state
  num_devices = jax.local_device_count()
  beta1_value = 1.0 - HPARAMS['one_minus_beta1']

  # 1. DEFINE THE MESH
  # We create a mesh of all available devices named 'batch'
  mesh = jax.sharding.Mesh(jax.devices(), ('batch',))
  replicated_spec = NamedSharding(mesh, P())
  data_sharding_spec = NamedSharding(mesh, P('batch'))
  fsdp_sharding_spec = NamedSharding(mesh, P('batch'))

  # 2. Dynamically choose sharding based on rank
  def get_sharding_spec(leaf):
    # If it's an array with dimensions, shard it.
    if hasattr(leaf, 'ndim') and leaf.ndim >= 1:
        return fsdp_sharding_spec
    # If it's a scalar (e.g., step count), replicate it.
    return replicated_spec

  # 3. Create Sharding Trees matching objects
  param_sharding_tree = jax.tree_util.tree_map(get_sharding_spec, current_param_container)
  opt_state_sharding_tree = jax.tree_util.tree_map(get_sharding_spec, optimizer_state)

  # 3. DISTRIBUTE DATA (device_put)
  # This moves data from CPU -> GPU and shards it immediately 
  batch = jax.device_put(batch, data_sharding_spec)
  current_param_container = jax.device_put(current_param_container, param_sharding_tree)
  optimizer_state = jax.device_put(optimizer_state, opt_state_sharding_tree)
    
  # Replicate small things
  model_state = jax.device_put(model_state, replicated_spec)
  rng = jax.device_put(rng, replicated_spec)

  # 4. COMPILE (JIT)
  # We compile the train_step to run on this specific Sharding setup.
  # out_shardings defines where the results land (keeping them sharded saves memory)
  jitted_train_step = jax.jit(
      train_step,
      static_argnums=(0, 1), # workload, opt_update_fn
      in_shardings=(
          replicated_spec,   # model_state
          opt_state_sharding_tree, # optimizer_state (Keep Sharded!)
          param_sharding_tree, # current_param_container (Keep Sharded!)
          data_sharding_spec, # batch
          replicated_spec,   # rng
          replicated_spec,   # grad_clip
          replicated_spec,   # label_smoothing
          replicated_spec,   # beta1
      ),
      out_shardings=(
          opt_state_sharding_tree, # new_optimizer_state
          param_sharding_tree, # updated_params
          replicated_spec,   # new_model_state
          replicated_spec,   # loss
          replicated_spec,   # grad_norm
      )
  )

  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
  else:
    grad_clip = None
  
  if hasattr(hyperparameters, 'label_smoothing'):
    label_smoothing = hyperparameters.label_smoothing
  else:
    label_smoothing = 0.0

  # 5. EXECUTE
  new_optimizer_state, new_params, new_model_state, loss, grad_norm = jitted_train_step(
      workload,
      opt_update_fn,
      model_state,
      optimizer_state,
      current_param_container,
      batch,
      rng,
      grad_clip,
      label_smoothing,
      beta1_value
  )

  # 6. RETURN STRATEGY
  # we keep optimizer_state sharded on GPU to save bandwidth/memory for next step.
  return (
      new_optimizer_state, 
      opt_update_fn
  ), jax.device_get(new_params), jax.device_get(new_model_state)


def get_batch_size(workload_name):
 if workload_name == 'criteo1tb':
   return 262_144
 elif workload_name == 'fastmri':
   return 16
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
   return 128 
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
    rng: spec.RandomState,
) -> Dict[str, spec.Tensor]:
  del workload, optimizer_state, current_param_container, model_state, hyperparameters, global_step, rng
  batch = next(input_queue)
  return batch
