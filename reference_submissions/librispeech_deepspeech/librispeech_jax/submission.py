"""Training algorithm track submission functions for LibriSpeech."""
import functools
from typing import Dict, Iterator, List, Tuple

from absl import logging
from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec

# import gc
# import torch

_GRAD_CLIP_EPS = 1e-6


def get_batch_size(workload_name):
  # Return the global batch size.
  del workload_name
  return 256


def optimizer(hyperparameters: spec.Hyperparameters, num_train_examples: int):
  opt_init_fn, opt_update_fn = optax.inject_hyperparams(optax.adamw)(
      b1=hyperparameters.beta1,
      b2=hyperparameters.beta2,
      eps=hyperparameters.epsilon,
      weight_decay=hyperparameters.weight_decay,
      learning_rate=0.0)
  return opt_init_fn, opt_update_fn


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del model_state
  del rng

  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  opt_init_fn, opt_update_fn = optimizer(hyperparameters,
                                         workload.num_train_examples)
  optimizer_state = opt_init_fn(params_zeros_like)
  return jax_utils.replicate(optimizer_state), opt_update_fn


def l2_regularization(params, l2_decay_rank_threshold):
  """Computes the squared l2 norm of the given parameters.

  This function will only filter for parameters with
  rank >= l2_decay_rank_threshold. So if this threshold is set to 2, then all
  1d (and lower) parameter arrays, including all bias and batch norm params,
  will be ignored in this computation.


  Args:
    params: Pytree containing parameters.
    l2_decay_rank_threshold: The calculation will only include parameters with
       param.ndim >= l2_decay_rank_threshold. Set to 2 to ignore all bias and
       batch_norm params in the model.

  Returns:
    weight_l2: the squared l2 norm of all params matching the threshold.
  """
  weight_penalty_params = jax.tree_leaves(params)
  weight_l2 = sum(
      jnp.sum(x**2)
      for x in weight_penalty_params
      if x.ndim >= l2_decay_rank_threshold)
  return weight_l2


def update_step(batch,
                params,
                batch_stats,
                optimizer_state,
                workload,
                global_step,
                hyperparameters,
                opt_update_fn,
                rng,
                lr):

  optimizer_state.hyperparams['learning_rate'] = lr

  def _loss_fn(params):
    """loss function used for training."""
    params_rng, dropout_rng = jax.random.split(rng, 2)
    (logits, logit_paddings), new_batch_stats = workload.model_fn(
        params,
        batch,
        batch_stats,
        spec.ForwardPassMode.TRAIN,
        {'params' : params_rng, 'dropout' : dropout_rng})

    loss = workload.loss_fn(batch['targets'], (logits, logit_paddings))
    return loss, new_batch_stats

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (loss, new_model_state), grad = grad_fn(params)

  grad_norm = jnp.sqrt(l2_regularization(grad, 0))

  loss, grad = lax.pmean((loss, grad), axis_name='batch')
  # with following gradient clipping code submission doesn't hang

  grad_clip = hyperparameters.grad_clip
  grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
  grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)

  grad = jax.tree_map(lambda x: x * grad_scaling_factor, grad)

  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state, params)
  updated_params = optax.apply_updates(params, updates)

  return updated_params, new_model_state, new_optimizer_state, loss, grad_norm


def update_params(workload,
                  current_param_container,
                  current_params_types,
                  model_state,
                  hyperparameters,
                  batch,
                  loss_type: spec.LossType,
                  optimizer_state,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng):
  """Return (updated_optimizer_state, updated_params)."""
  del eval_results
  del loss_type
  del current_params_types

  lr = workload.get_learning_rate(global_step, hyperparameters)
  optimizer_state, opt_update_fn = optimizer_state

  update_fn = functools.partial(
      update_step,
      opt_update_fn=opt_update_fn,
      global_step=global_step,
      workload=workload,
      hyperparameters=hyperparameters,
      rng=rng,
      lr=lr)

  pmapped_update_step = jax.pmap(
      update_fn, axis_name='batch', in_axes=(0, 0, 0, 0))
  new_params, new_model_state, new_optimizer_state, loss, grad_norm = pmapped_update_step(  # pylint: disable=line-too-long
    batch,
    current_param_container,
    model_state,
    optimizer_state)

  logging.info('%d) loss = %0.03f, grad_norm = %0.03f',
               global_step,
               loss.mean(),
               grad_norm.mean())

  return (new_optimizer_state, opt_update_fn), new_params, new_model_state


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
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
  del optimizer_state
  del current_param_container
  del global_step
  del rng
  del hyperparameters
  del workload
  return next(input_queue)
