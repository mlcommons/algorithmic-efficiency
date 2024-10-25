"""Training algorithm track submission functions for CIFAR10."""

import functools
from typing import Any, Dict, Iterator, List, Optional, Tuple

from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec


def get_batch_size(workload_name):
  # Return the global batch size.
  batch_sizes = {'cifar': 128}
  return batch_sizes[workload_name]


def cosine_decay(lr, step, total_steps):
  ratio = jnp.maximum(0., step / total_steps)
  mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
  return mult * lr


def create_learning_rate_fn(hparams: spec.Hyperparameters,
                            steps_per_epoch: int):
  """Create learning rate schedule."""
  base_learning_rate = hparams.learning_rate * get_batch_size('cifar') / 128.
  warmup_fn = optax.linear_schedule(
      init_value=0.,
      end_value=base_learning_rate,
      transition_steps=hparams.warmup_epochs * steps_per_epoch)
  cosine_epochs = max(hparams.num_epochs - hparams.warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[hparams.warmup_epochs * steps_per_epoch])
  return schedule_fn


def optimizer(hyperparameters: spec.Hyperparameters, num_train_examples: int):
  steps_per_epoch = num_train_examples // get_batch_size('cifar')
  learning_rate_fn = create_learning_rate_fn(hyperparameters, steps_per_epoch)
  opt_init_fn, opt_update_fn = optax.sgd(
      nesterov=True,
      momentum=hyperparameters.momentum,
      learning_rate=learning_rate_fn)
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
    in_axes=(None, None, 0, 0, 0, None, 0, 0),
    static_broadcasted_argnums=(0, 1))
def pmapped_train_step(workload,
                       opt_update_fn,
                       model_state,
                       optimizer_state,
                       current_param_container,
                       hyperparameters,
                       batch,
                       rng):

  def _loss_fn(params):
    """loss function used for training."""
    logits, new_model_state = workload.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True)
    loss_dict = workload.loss_fn(batch['targets'], logits)
    loss = loss_dict['summed'] / loss_dict['n_valid_examples']
    weight_penalty_params = jax.tree_util.tree_leaves(params)
    weight_l2 = sum(jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1)
    weight_penalty = hyperparameters.l2 * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, new_model_state

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (_, new_model_state), grad = grad_fn(current_param_container)
  grad = lax.pmean(grad, axis_name='batch')
  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
  updated_params = optax.apply_updates(current_param_container, updates)
  return new_optimizer_state, updated_params, new_model_state


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
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
                  rng: spec.RandomState,
                  train_state: Optional[Dict[str, Any]] = None
                  ) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del global_step
  del train_state
  del eval_results
  optimizer_state, opt_update_fn = optimizer_state
  per_device_rngs = jax.random.split(rng, jax.local_device_count())
  new_optimizer_state, new_params, new_model_state = pmapped_train_step(
      workload, opt_update_fn, model_state, optimizer_state,
      current_param_container, hyperparameters, batch, per_device_rngs)
  return (new_optimizer_state, opt_update_fn), new_params, new_model_state


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
