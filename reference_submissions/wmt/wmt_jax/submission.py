"""Training algorithm track submission functions for WMT."""

import functools
from typing import Dict, Iterator, List, Tuple

from flax import jax_utils
import jax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec


def get_batch_size(workload_name):
  batch_sizes = {'wmt': 128}
  return batch_sizes[workload_name]


def create_learning_rate_scheduler(
    factors='constant * linear_warmup * rsqrt_decay',
    base_learning_rate=0.5,
    warmup_steps=1000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: string, factors separated by "*" that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: int, how many steps to warm up for in the warmup schedule.
    decay_factor: float, the amount to decay the learning rate by.
    steps_per_decay: int, how often to decay the learning rate.
    steps_per_cycle: int, steps per cycle when using cosine decay.

  Returns:
    a function learning_rate(step): float -> {"learning_rate": float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == 'cosine_decay':
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError(f'Unknown factor {name}.')
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del model_params
  del model_state
  del rng
  learning_rate_fn = create_learning_rate_scheduler(
      base_learning_rate=hyperparameters.learning_rate, warmup_steps=1000)
  opt_init_fn, opt_update_fn = optax.adam(
      b1=1.0 - hyperparameters.one_minus_beta_1,
      b2=0.98,
      eps=hyperparameters.epsilon,
      learning_rate=learning_rate_fn)
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  optimizer_state = opt_init_fn(params_zeros_like)
  return jax_utils.replicate(optimizer_state), opt_update_fn


@functools.partial(
    jax.pmap,
    in_axes=(None, None, 0, 0, 0, 0, None),
    axis_name='batch',
    static_broadcasted_argnums=(0, 1, 6))
def pmapped_train_step(workload,
                       opt_update_fn,
                       optimizer_state,
                       current_param_container,
                       batch,
                       dropout_rng,
                       hyperparameters):
  """Perform a single training step."""
  if hasattr(hyperparameters, 'dropout_rate'):
    dropout_rate = hyperparameters.dropout_rate
  else:
    dropout_rate = 0.1
  if hasattr(hyperparameters, 'attention_dropout_rate'):
    attention_dropout_rate = hyperparameters.attention_dropout_rate
  else:
    attention_dropout_rate = 0.1

  def _loss_fn(params):
    """Loss function used for training."""
    logits, _ = workload.model_fn(
        params,
        batch,
        model_state=None,
        mode=spec.ForwardPassMode.TRAIN,
        rng=dropout_rng,
        dropout_rate=dropout_rate,
        aux_dropout_rate=attention_dropout_rate,
        update_batch_norm=False)
    targets = batch['targets']
    weights = jnp.where(targets > 0, 1.0, 0.0)
    loss = (workload.loss_fn(targets, logits, label_smoothing=0.1) *
            weights).sum() / weights.sum()
    return loss

  grad_fn = jax.value_and_grad(_loss_fn)
  _, grad = grad_fn(current_param_container)
  grad = jax.lax.pmean(grad, axis_name='batch')
  updates, new_optimizer_state = opt_update_fn(
      grad, optimizer_state, current_param_container)
  updated_params = optax.apply_updates(current_param_container, updates)
  return new_optimizer_state, updated_params


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
    rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params)."""
  del current_params_types
  del eval_results
  del global_step
  del model_state
  del loss_type

  optimizer_state, opt_update_fn = optimizer_state
  dropout_rngs = jax.random.split(rng, jax.local_device_count())
  new_optimizer_state, updated_params = pmapped_train_step(
      workload,
      opt_update_fn,
      optimizer_state,
      current_param_container,
      batch,
      dropout_rngs,
      hyperparameters)
  return (new_optimizer_state, opt_update_fn), updated_params, None


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
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
  return next(input_queue)
