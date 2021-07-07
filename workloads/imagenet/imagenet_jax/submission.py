"""Training algorithm track submission functions for ImageNet."""

import functools
from typing import Iterator, List, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils
from jax import lax


import spec



def get_batch_size(workload_name):
  del workload_name
  return 256


def cosine_decay(lr, step, total_steps):
  ratio = jnp.maximum(0., step / total_steps)
  mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
  return mult * lr


def create_learning_rate_schedule(hparams, max_training_steps):
  """Polynomial learning rate schedule for LARS optimizer.
  This function is copied from
  https://github.com/google/init2winit/blob/master/init2winit/schedules.py
  Args:
    hparams: Relevant hparams are base_lr, warmup_steps.
    max_training_steps: Used to calculate the number of decay steps.
  Returns:
    lr_fn: A function mapping global_step to lr.
  """
  hparams = {
    'base_lr': 7.05/64,
    'warmup_power': 2.0,
    'warmup_steps': 706.5*100,
    'end_lr': 0.000006,
    'decay_end': 2512*100,
    'power': 2.0,
    'start_lr': 0.0,
  }
  decay_steps = max_training_steps - hparams['warmup_steps'] + 1
  def step_fn(step):
    step = jax.lax.cond(
      (lambda step, decay_end :
        decay_end > 0 and step >= decay_end)(step, hparams['decay_end']),
      lambda _: hparams['decay_end'],
      lambda _: step,
      operand=None)
    r = (step / hparams['warmup_steps']) ** hparams['warmup_power']
    warmup_lr = (
        hparams['base_lr'] * r + (1 - r) * hparams['start_lr'])
    decay_step = jnp.minimum(step - hparams['warmup_steps'], decay_steps)
    poly_lr = (
        hparams['end_lr'] + (hparams['base_lr'] - hparams['end_lr']) *
        (1 - decay_step / decay_steps) ** hparams['power'])
    return jnp.where(step <= hparams['warmup_steps'], warmup_lr, poly_lr)
  return step_fn


def optimizer(hyperparameters: spec.Hyperparamters, learning_rate_fn):
  opt_init_fn, opt_update_fn = optax.sgd(
      nesterov=True,
      learning_rate=learning_rate_fn
    )
  return opt_init_fn, opt_update_fn


def init_optimizer_state(
    workload: spec.Workload,
    model_params: spec.ParameterTree,
    model_state: spec.ModelAuxillaryState,
    hyperparameters: spec.Hyperparamters,
    rng: spec.RandomState) -> spec.OptimizerState:
  workload.learning_rate_fn = create_learning_rate_schedule(hyperparameters, workload.steps_per_epoch)
  params_zeros_like = jax.tree_map(
      lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes)
  opt_init_fn, _ = optimizer(hyperparameters, workload.learning_rate_fn)
  optimizer_state = opt_init_fn(params_zeros_like)
  return jax_utils.replicate(optimizer_state)


# We need to jax.pmap here instead of inside update_params because the latter
# the latter would recompile the function every step.
@functools.partial(
  jax.pmap,
  axis_name='batch',
  in_axes=(None, 0, 0, 0, None, None, 0, None),
  static_broadcasted_argnums=(0,))
def pmapped_train_step(workload, model_state, optimizer_state, current_params,
                       step, hyperparameters, batch, rng):
  def _loss_fn(params):
    """loss function used for training."""
    variables = {'params': params, **model_state}
    logits, new_model_state = workload.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True)
    loss = workload.loss_fn(batch['label'], logits)
    weight_penalty_params = jax.tree_leaves(variables['params'])
    weight_decay = 0.0001
    weight_l2 = sum([jnp.sum(x ** 2)
                    for x in weight_penalty_params
                    if x.ndim > 1])
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_model_state, logits)

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  aux, grad = grad_fn(current_params)
  grad = lax.pmean(grad, axis_name='batch')
  _, opt_update_fn = optimizer(hyperparameters, workload.learning_rate_fn)
  new_model_state, logits = aux[1]
  updates, new_optimizer_state = opt_update_fn(
      grad, optimizer_state, current_params)
  updated_params = optax.apply_updates(current_params, updates)
  metrics = workload.compute_metrics(logits, batch['label'])

  return new_model_state, new_optimizer_state, updated_params, metrics


def update_params(
    workload: spec.Workload,
    current_params: spec.ParameterTree,
    current_params_types: spec.ParameterTypeTree,
    model_state: spec.ModelAuxillaryState,
    hyperparameters: spec.Hyperparamters,
    augmented_and_preprocessed_input_batch: spec.Tensor,
    label_batch: spec.Tensor,
    loss_type: spec.LossType,
    optimizer_state: spec.OptimizerState,
    eval_results: List[Tuple[int, float]],
    global_step: int,
    rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  batch = {
    'image': augmented_and_preprocessed_input_batch,
    'label': label_batch
  }
  new_model_state, new_optimizer, new_params, metrics = pmapped_train_step(
    workload, model_state, optimizer_state, current_params, global_step,
    hyperparameters, batch, rng)

  workload.epoch_metrics.append(metrics)

  if global_step % workload.steps_per_epoch == 0:
    # sync batch statistics across replicas once per epoch
    new_model_state = workload.sync_batch_stats(new_model_state)

  return new_optimizer, new_params, new_model_state


def data_selection(
    workload: spec.Workload,
    input_queue: Iterator[Tuple[spec.Tensor, spec.Tensor]],
    optimizer_state: spec.OptimizerState,
    current_params: spec.ParameterTree,
    hyperparameters: spec.Hyperparamters,
    global_step: int,
    rng: spec.RandomState) -> Tuple[spec.Tensor, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a single training example and label.

  Return a tuple of input label batches.
  """
  x = next(input_queue)
  return x['image'], x['label']
