"""Training algorithm track submission functions for FastMRI in Jax."""

import functools
from typing import Dict, Iterator, List, Tuple

from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec


def get_batch_size(workload_name):
  # Return the global batch size.
  del workload_name
  return 64


def create_learning_rate_fn(hparams: spec.Hyperparameters,
                            steps_per_epoch: int):
  """Create learning rate schedule."""
  max_num_train_steps = 500 * steps_per_epoch
  decay_epoch_period = hparams.lr_step_size * steps_per_epoch
  decay_events = range(decay_epoch_period,
                       max_num_train_steps,
                       decay_epoch_period)
  schedule_fn = optax.piecewise_constant_schedule(
      init_value=hparams.learning_rate,
      boundaries_and_scales={t: hparams.lr_gamma for t in decay_events})
  return schedule_fn


def optimizer(hyperparameters: spec.Hyperparameters, num_train_examples: int):
  steps_per_epoch = num_train_examples // get_batch_size('imagenet_resnet')
  learning_rate_fn = create_learning_rate_fn(hyperparameters, steps_per_epoch)
  opt_init_fn, opt_update_fn = optax.rmsprop(
      learning_rate=learning_rate_fn,
      decay=0.99)
  return opt_init_fn, opt_update_fn


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del model_params
  del model_state
  del rng
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  opt_init_fn, opt_update_fn = optimizer(hyperparameters,
                                         workload.num_train_examples)
  optimizer_state = opt_init_fn(params_zeros_like)
  return jax_utils.replicate(optimizer_state), opt_update_fn


@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, None, 0, 0, None, 0, 0),
    static_broadcasted_argnums=(0, 1, 4))
def pmapped_train_step(workload,
                       opt_update_fn,
                       optimizer_state,
                       current_param_container,
                       hyperparameters,
                       batch,
                       rng):

  def _loss_fn(params):
    """loss function used for training."""
    logits, _ = workload.model_fn(
        params,
        batch,
        model_state=None,
        mode=spec.ForwardPassMode.TRAIN,
        rng=rng,
        update_batch_norm=True)
    loss = jnp.mean(workload.loss_fn(batch['targets'], logits))
    weight_penalty_params = jax.tree_leaves(params)
    weight_l2 = sum(jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1)
    weight_penalty = hyperparameters.l2 * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss

  grad_fn = jax.grad(_loss_fn)
  grad = grad_fn(current_param_container)
  grad = lax.pmean(grad, axis_name='batch')
  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
  updated_params = optax.apply_updates(current_param_container, updates)

  return new_optimizer_state, updated_params


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
  del model_state
  del loss_type
  del eval_results
  del global_step

  optimizer_state, opt_update_fn = optimizer_state
  per_device_rngs = jax.random.split(rng, jax.local_device_count())
  new_optimizer_state, new_params = pmapped_train_step(
      workload, opt_update_fn, optimizer_state,
      current_param_container, hyperparameters, batch, per_device_rngs)
  return (new_optimizer_state, opt_update_fn), new_params, None


def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a batch of training examples and labels.
  """
  del workload
  del optimizer_state
  del current_param_container
  del hyperparameters
  del global_step
  del rng
  return next(input_queue)
