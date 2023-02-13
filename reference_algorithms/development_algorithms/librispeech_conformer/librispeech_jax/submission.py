"""Training algorithm track submission functions for LibriSpeech."""
import functools
from typing import Dict, Iterator, List, Tuple

from absl import logging
from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax

from algorithmic_efficiency import spec

_GRAD_CLIP_EPS = 1e-6


def get_batch_size(workload_name):
  # Return the global batch size.
  del workload_name
  return 256


def get_learning_rate(step, hyperparams):
  warmup_steps = hyperparams.warmup_steps
  if step < warmup_steps:
    current_lr = (step * hyperparams.base_lr) / warmup_steps
  else:
    decay_factor = (1 + np.cos(step / hyperparams.training_steps * np.pi)) * 0.5
    current_lr = hyperparams.base_lr * decay_factor
  return current_lr


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
  weight_penalty_params = jax.tree_util.tree_leaves(params)
  weight_l2 = sum(
      jnp.sum(x**2)
      for x in weight_penalty_params
      if x.ndim >= l2_decay_rank_threshold)
  return weight_l2


@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, None, 0, 0, 0, None, 0, 0, None),
    static_broadcasted_argnums=(0, 1))
def pmapped_train_step(workload,
                       opt_update_fn,
                       model_state,
                       optimizer_state,
                       current_param_container,
                       hyperparameters,
                       batch,
                       rng,
                       lr):
  optimizer_state.hyperparams['learning_rate'] = lr

  def _loss_fn(params):
    """loss function used for training."""
    (logits, logit_paddings), new_model_state = workload.model_fn(
        params,
        batch,
        model_state,
        mode=spec.ForwardPassMode.TRAIN,
        rng=rng,
        update_batch_norm=True)
    loss_dict = workload.loss_fn(batch['targets'], (logits, logit_paddings))
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

  grad_clip = hyperparameters.grad_clip
  grad_norm = jnp.sqrt(l2_regularization(grad, 0))
  scaled_grad = jax.tree_map(
      lambda x: x / (grad_norm + _GRAD_CLIP_EPS) * grad_clip, grad)
  grad = jax.lax.cond(grad_norm > grad_clip,
                      lambda _: scaled_grad,
                      lambda _: grad,
                      None)

  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
  updated_params = optax.apply_updates(current_param_container, updates)

  return new_model_state, new_optimizer_state, updated_params, loss, grad_norm


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
  """Return (updated_optimizer_state, updated_params)."""
  del current_params_types
  del eval_results
  del loss_type

  lr = get_learning_rate(global_step, hyperparameters)
  optimizer_state, opt_update_fn = optimizer_state
  per_device_rngs = jax.random.split(rng, jax.local_device_count())
  outputs = pmapped_train_step(workload,
                               opt_update_fn,
                               model_state,
                               optimizer_state,
                               current_param_container,
                               hyperparameters,
                               batch,
                               per_device_rngs,
                               lr)
  new_model_state, new_optimizer_state, new_params, loss, grad_norm = outputs

  if global_step <= 1000 or global_step % 100 == 0:
    logging.info('%d) loss = %0.3f, grad_norm = %0.3f lr = %0.6f',
                 global_step,
                 loss.mean(),
                 grad_norm.mean(),
                 lr)

    if workload.metrics_logger is not None:
      workload.metrics_logger.append_scalar_metrics(
          {
              'train_step_ctc_loss': loss.mean(),
              'grad_norm': grad_norm.mean(),
              'learning_rate': lr,
          },
          global_step)

  return (new_optimizer_state, opt_update_fn), new_params, new_model_state


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
