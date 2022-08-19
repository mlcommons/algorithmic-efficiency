"""Training algorithm track submission functions for LibriSpeech."""
from typing import Dict, Iterator, List, Tuple

import jax.numpy as jnp
import jax
import flax 
import optax
from flax import jax_utils
import functools
import jax.lax as lax
import numpy as np
from absl import logging

from algorithmic_efficiency import spec

_GRAD_CLIP_EPS = 1e-6

def get_batch_size(workload_name):
  # Return the global batch size.
  del workload_name
  return 16


def create_learning_rate_fn(hparams: spec.Hyperparameters):
  """Create learning rate schedule."""
  base_learning_rate = hparams.base_lr
  warmup_fn = optax.linear_schedule(
      init_value=0.,
      end_value=base_learning_rate,
      transition_steps=hparams.warmup_steps)
  cosine_steps = max(hparams.num_training_steps - hparams.warmup_steps, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_steps)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[hparams.warmup_steps])
  return schedule_fn

def optimizer(hyperparameters: spec.Hyperparameters, num_train_examples: int):
  learning_rate_fn = create_learning_rate_fn(hyperparameters)
  opt_init_fn, opt_update_fn = optax.adamw(
      b1=hyperparameters.beta1,
      b2=hyperparameters.beta2,
      eps=hyperparameters.epsilon,
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
    variables = {'params': params, **model_state}
    params_rng, dropout_rng = jax.random.split(rng, 2)
    (logits, logit_paddings), new_model_state = workload.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        {'params' : params_rng, 'dropout' : dropout_rng},
        update_batch_norm=True)

    loss = jnp.mean(workload.loss_fn(logits, logit_paddings, batch['targets'], batch['target_paddings']))
    return loss, new_model_state

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (loss, new_model_state), grad = grad_fn(current_param_container)
  grad = lax.pmean(grad, axis_name='batch')

  grad_clip = hyperparameters.grad_clip
  grad_norm = jnp.sqrt(sum([
      jnp.sum(x**2)
      for x in jax.tree_leaves(current_param_container)
  ]))
  scaled_grad = jax.tree_map(
      lambda x: x / (grad_norm + _GRAD_CLIP_EPS) * grad_clip, grad)
  grad = jax.lax.cond(grad_norm > grad_clip, lambda _: scaled_grad,
                      lambda _: grad, None)
  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
  updated_params = optax.apply_updates(current_param_container, updates)

  return new_model_state, new_optimizer_state, updated_params, loss


def update_params(
    workload: spec.Workload,
    current_param_container: spec.ParameterContainer,
    current_params_types: spec.ParameterTypeTree,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparameters,
    batch: Dict[str, spec.Tensor],
    # This will define the output activation via `output_activation_fn`.
    loss_type: spec.LossType,
    optimizer_state: spec.OptimizerState,
    eval_results: List[Tuple[int, float]],
    global_step: int,
    rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params)."""
  del current_params_types
  del eval_results
  del global_step
  del loss_type

  optimizer_state, opt_update_fn = optimizer_state
  per_device_rngs = jax.random.split(rng, jax.local_device_count())
  new_model_state, new_optimizer_state, new_params, loss = pmapped_train_step(
      workload, opt_update_fn, model_state, optimizer_state,
      current_param_container, hyperparameters, batch, per_device_rngs)
  logging.info('loss = {}'.format(loss))
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
