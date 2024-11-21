"""Submission file for a SGD with Nesterov momentum optimizer in Jax."""

import functools
from typing import Callable, Dict, Iterator, List, Tuple

from flax import jax_utils
import jax
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec, sharding_utils

_GRAD_CLIP_EPS = 1e-6


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a Nesterov optimizer and a learning rate schedule."""
  del model_params
  del model_state
  del rng

  # Create learning rate schedule.
  lr_schedule_fn = create_lr_schedule_fn(workload.step_hint, hyperparameters)

  # Create optimizer.
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  opt_init_fn, opt_update_fn = sgd(
      learning_rate=lr_schedule_fn,
      weight_decay=hyperparameters.weight_decay,
      momentum=1.0 - hyperparameters.one_minus_beta1,
      nesterov=True)
  optimizer_state = opt_init_fn(params_zeros_like)

  return optimizer_state, opt_update_fn


def create_lr_schedule_fn(
    step_hint: int,
    hyperparameters: spec.Hyperparameters) -> Callable[[int], float]:
  warmup_steps = int(hyperparameters.warmup_factor * step_hint)
  warmup_fn = optax.linear_schedule(
      init_value=0.,
      end_value=hyperparameters.learning_rate,
      transition_steps=warmup_steps)
  decay_steps = step_hint - warmup_steps
  polynomial_schedule_fn = optax.polynomial_schedule(
      init_value=hyperparameters.learning_rate,
      end_value=hyperparameters.learning_rate * hyperparameters.end_factor,
      power=1,
      transition_steps=int(decay_steps * hyperparameters.decay_steps_factor))
  lr_schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, polynomial_schedule_fn], boundaries=[warmup_steps])
  return lr_schedule_fn


# Forked from github.com/google/init2winit/blob/master/init2winit/ (cont. below)
# optimizer_lib/optimizers.py.
def sgd(learning_rate, weight_decay, momentum=None, nesterov=False):
  r"""A customizable gradient descent optimizer.

  NOTE: We apply weight decay **before** computing the momentum update.
  This is equivalent to applying WD after for heavy-ball momentum,
  but slightly different when using Nesterov acceleration. This is the same as
  how the Flax optimizers handle weight decay
  https://flax.readthedocs.io/en/latest/_modules/flax/optim/momentum.html.

  Args:
    learning_rate: The learning rate. Expected as the positive learning rate,
      for example `\alpha` in `w -= \alpha * u` (as opposed to `\alpha`).
    weight_decay: The weight decay hyperparameter.
    momentum: The momentum hyperparameter.
    nesterov: Whether or not to use Nesterov momentum.

  Returns:
    An optax gradient transformation that applies weight decay and then one of a
    {SGD, Momentum, Nesterov} update.
  """
  return optax.chain(
      optax.add_decayed_weights(weight_decay),
      optax.sgd(
          learning_rate=learning_rate, momentum=momentum, nesterov=nesterov))


# @functools.partial(
#     jax.pmap,
#     axis_name='batch',
#     in_axes=(None, None, 0, 0, 0, 0, 0, None, None),
#     static_broadcasted_argnums=(0, 1),
#     donate_argnums=(2, 3, 4))
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
  # # Get correct global mean loss and grad.
  loss = summed_loss / n_valid_examples
  grad = jax.tree_map(lambda x: x / n_valid_examples, grad)
  grad_norm = jnp.sqrt(
      sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

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

  mesh = sharding_utils.get_mesh()
  # Create shardings for each argument
  replicated = NamedSharding(mesh, P())  # No partitioning
  sharded = NamedSharding(mesh, P('batch'))  # Partition along batch dimension

  # Create the sharding rules for each argument
  arg_shardings = (
      # workload is static
      # opt_update_fn is static
      replicated,     # model_state
      replicated,     # optimizer_state
      replicated,     # current_param_container
      sharded,     # batch
      sharded,     # per_device_rngs
      replicated,  # grad_clip
      replicated   # label_smoothing
  )
  out_shardings = (
      replicated,               # new_optimizer_state
      replicated,               # updated_params
      replicated,               # new_model_state
      replicated,               # loss
      replicated                # grad_norm
  )
  # Jit with shardings
  jitted_train_step = jax.jit(
      train_step,
      static_argnums=(0, 1),
      donate_argnums=(2, 3, 4),
      in_shardings=arg_shardings,
      out_shardings=out_shardings
  )
  outputs = jitted_train_step(workload,
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
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
        }, global_step)
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
