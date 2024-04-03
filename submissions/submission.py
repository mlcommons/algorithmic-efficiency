"""Submission file for an NAdamW optimizer with warmup+cosine LR in Jax."""

import functools

# isort: off
# We have to turn off isort here to resolve a conflict between isort and yapf.
from typing import (Any,
                    Callable,
                    Dict,
                    Iterator,
                    List,
                    NamedTuple,
                    Optional,
                    Tuple,
                    Union)
# isort: on

from absl import logging
import chex
from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
import optax
import math
from flax.training import checkpoints as flax_checkpoints
from typing import Sequence

from algorithmic_efficiency import spec

_GRAD_CLIP_EPS = 1e-6


# Make sure the traning horizons for all points add up to 3
# since they are w.r.t. the external tuning stephint
HPARAMS = [
  {
    "dropout_rate": 0.1,
    "learning_rate": 0.0014271957958295392,
    "one_minus_beta1": 0.03380478752,
    "beta2": 0.9957304053273589,
    "weight_decay": 0.09153141484048229,
    "warmup_factor": 0.01,
    "label_smoothing": 0.1,
    # Debug criteo
    "training_horizon": 0.0006,
    # "training_horizon": 1,
},
           {
               "dropout_rate": 0.0,
               "learning_rate": 0.001768509931943289,
               "one_minus_beta1": 0.05850208614,
               "beta2": 0.9768053375036079,
               "weight_decay": 0.0279513959224539,
               "warmup_factor": 0.02,
               "label_smoothing": 0.2,
               "training_horizon": 1,
               # Debug criteo
               "training_horizon": 0.0006,
               # "training_horizon": 1,
           },
           {
               "dropout_rate": 0.1,
               "learning_rate": 0.0023792566965593815,
               "one_minus_beta1": 0.01990335215,
               "beta2": 0.9632738717172477,
               "weight_decay": 0.3417568278549717,
               "warmup_factor": 0.01,
               # Debug criteo
               "training_horizon": 0.0006,
              #  "training_horizon": 0.75
           }
           ]


def replicate_checkpoint(latest: dict,
                         pytree_keys: Sequence[str],
                         replicate: bool = True) -> dict:
  """Restores from the provided checkpoint.

  Args:
    latest: A dict representing the state of the
      checkpoint we want to restore.
    pytree_keys: A sequence of keys into `latest` that are pytrees, which will
      be replicated if replicate=True.
    replicate: If set, replicate the state across devices.

  Returns:
    A JAX pytree holding the arrays that need to be replicated/unreplicated.
  """
  pytree = {k: latest[k] for k in pytree_keys}
  if replicate:
    pytree = jax_utils.replicate(pytree)
  extra_dict = {k: latest[k] for k in latest.keys() if k not in pytree_keys}
  pytree.update(extra_dict)
  return pytree


# Forked from
# github.com/google/init2winit/blob/master/init2winit/optimizer_lib/alias.py
def nadamw(
    learning_rate: Union[float, optax.Schedule],
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[optax.Params],
                                                    Any]]] = None,
) -> optax.GradientTransformation:
  """Rescale updates according to the NAdam algorithm.

  References:
  There seem to be multiple versions of NAdam. The original version is here
  https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ (the official PyTorch
  implementation also follows this).
  Current code implements a simpler version with no momentum decay and slightly
  different bias correction terms. The exact description can be found here
  https://arxiv.org/pdf/1910.05446.pdf (Table 1).

  Args:
    learning_rate: A fixed global scaling factor.
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: Whether to use bias correction.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent with
      other frameworks such as PyTorch, but different from (Loshchilov et al,
      2019) where the weight decay is only multiplied with the "schedule
      multiplier", but not the base learning rate.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Nadam gradient transformations are applied to all parameters.

  Returns:
    An (init_fn, update_fn) tuple.
  """
  return optax.chain(
      scale_by_nadam(b1, b2, eps, eps_root, debias),
      optax.add_decayed_weights(weight_decay, weight_decay_mask),
      scale_by_learning_rate(learning_rate))


# All functions below are forked from
# github.com/google/init2winit/blob/master/init2winit/optimizer_lib/transform.py
def scale_by_nadam(b1: float = 0.9,
                   b2: float = 0.999,
                   eps: float = 1e-8,
                   eps_root: float = 0.0,
                   debias: bool = True,
                   power: float = 0.5) -> optax.GradientTransformation:
  """Rescale updates according to the NAdam algorithm.

  References:
  There seem to be multiple versions of NAdam. The original version is here
  https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ (the pytorch imp. also
  follows this).

  Current code implements a simpler version with no momentum decay and slightly
  different (standard Adam) bias correction terms. The exact description can be
  found here https://arxiv.org/pdf/1910.05446.pdf (Table 1)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: Whether to use bias correction.
    power: The power to use in the preconditioner (0.5 in default adam).
  Returns:
    An (init_fn, update_fn) tuple.
  """
  raise_power = jnp.sqrt if power == 0.5 else lambda x: jnp.power(x, power)

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = _update_moment(updates, mu, b1, 1)
    mu_hat = mu_hat if not debias else _bias_correction(mu_hat, b1, count)
    nu_hat = nu if not debias else _bias_correction(nu, b2, count)
    updates = jax.tree_map(
        lambda m, v: m / (raise_power(v + eps_root) + eps), mu_hat, nu_hat)
    return updates, ScaleByAdamState(count=count, mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


class ScaleByAdamState(NamedTuple):
  """State for the NAdam algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: optax.Updates
  nu: optax.Updates


def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_map(
      lambda g, t: (1 - decay) * (g**order) + decay * t, updates, moments)


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)


def scale_by_learning_rate(learning_rate, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a NAdamW optimizer and a learning rate schedule."""
  del model_state
  del rng
  del hyperparameters

  optimizer_state = {'optimizers': []}
  optimizer_state['hyperparameter_points'] = HPARAMS
  optimizer_state['lr_fns'] = []

  def jax_cosine_warmup(step_hint: int, hyperparameters):
    # Create learning rate schedule.
    warmup_steps = int(hyperparameters['warmup_factor'] * step_hint)
    warmup_fn = optax.linear_schedule(
        init_value=0.,
        end_value=hyperparameters['learning_rate'],
        transition_steps=warmup_steps)
    cosine_steps = max(step_hint - warmup_steps, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=hyperparameters['learning_rate'], decay_steps=cosine_steps)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps])
    return schedule_fn

  # Create optimizer + LR schedule.
  end_step = 0
  for hyperparameters in optimizer_state['hyperparameter_points']:
    horizon_steps = math.ceil(hyperparameters['training_horizon'] *
                              workload.step_hint)
    end_step = end_step + horizon_steps
    lr_schedule_fn = jax_cosine_warmup(horizon_steps, hyperparameters)
    opt_init_fn, opt_update_fn = nadamw(
        learning_rate=lr_schedule_fn,
        b1=1.0 - hyperparameters['one_minus_beta1'],
        b2=hyperparameters['beta2'],
        eps=1e-8,
        weight_decay=hyperparameters['weight_decay'])
    # Todo remove sub_optimizer_state 
    # sub_optimizer_state = opt_init_fn(params_zeros_like)
    optimizer_state['optimizers'].append(
        (end_step, opt_init_fn, opt_update_fn))
    optimizer_state['lr_fns'].append(lr_schedule_fn)
    optimizer_state['index'] = 0

  # Initialize first optstate
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                     workload.param_shapes)
  _, opt_init_fn, _, = optimizer_state['optimizers'][0]
  optimizer_state['current_opt_state'] = opt_init_fn(params_zeros_like)

  # Save initial model weights
  model_params = jax.device_get(model_params)
  checkpoint_state = {'model_params': model_params}
  flax_checkpoints.save_checkpoint(
      '/tmp', target=checkpoint_state, step=0, overwrite=True, keep=1)

  return optimizer_state, None


@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, None, 0, 0, 0, 0, 0, None, None),
    static_broadcasted_argnums=(0, 1),
    # todo add donate argnum 3
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
  del hyperparameters

  # End step of the current point
  optimizer_state, _ = optimizer_state # maybe_restore_from_checkpoint call forces optimizer state to be tuple
  horizon_end_step, _, opt_update_fn = optimizer_state['optimizers'][optimizer_state['index']]
  current_opt_state = optimizer_state['current_opt_state']

  # If we have reached the end of the current opt point horizon progress the index
  if global_step == horizon_end_step:
    # Reset model weights
    logging.info('Moving to next opt point.')
    checkpoint_state = {
        'model_params': jax_utils.unreplicate(current_param_container)
    }
    ckpt = flax_checkpoints.restore_checkpoint('/tmp/', target=checkpoint_state)
    current_param_container = ckpt['model_params']
    optimizer_state['index'] += 1
    try:
      horizon_end_step, opt_init_fn, opt_update_fn = optimizer_state['optimizers'][
          optimizer_state['index']]
    except IndexError:
      raise spec.TrainingCompleteError
    
    # Initialize new opt_state
    params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                  workload.param_shapes)
    optimizer_state['current_opt_state'] = opt_init_fn(params_zeros_like)
    current_opt_state = optimizer_state['current_opt_state']

  # Check for label_smoothing and grad_clip
  hyperparameters = optimizer_state['hyperparameter_points'][optimizer_state['index']]

  if hasattr(hyperparameters, 'label_smoothing'):
    label_smoothing = hyperparameters['label_smoothing']
  else:
    label_smoothing = 0.0
  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters['grad_clip']
  else:
    grad_clip = None

  # Pmapped update step
  per_device_rngs = jax.random.split(rng, jax.local_device_count())
  outputs = pmapped_train_step(workload,
                               opt_update_fn,
                               model_state,
                               current_opt_state,
                               current_param_container,
                               batch,
                               per_device_rngs,
                               grad_clip,
                               label_smoothing)
  new_current_opt_state, new_params, new_model_state, loss, grad_norm = outputs

  optimizer_state['current_opt_state'] = new_current_opt_state

  # Log loss, grad_norm.
  if global_step % 100 == 0 and workload.metrics_logger is not None:
    # lr_fn = optimizer_state['lr_fns'][optimizer_state['index']]
    workload.metrics_logger.append_scalar_metrics(
        {
            'loss': loss[0],
            'grad_norm': grad_norm[0],
            # 'lr': lr_fn(sub_optimizer_state[-1].count)[0]
        },
        global_step)

  # The maybe_restore_from_checkpoint call in submission_runner expects a tuple for
  # optimizer state.
  return (optimizer_state, None), new_params, new_model_state


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
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
