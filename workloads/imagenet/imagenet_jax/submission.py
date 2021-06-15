"""Training algorithm track submission functions for ImageNet."""

import functools
from typing import Iterator, List, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils

import spec



def get_batch_size(workload_name):
  del workload_name
  return 128


def cosine_decay(lr, step, total_steps):
  ratio = jnp.maximum(0., step / total_steps)
  mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
  return mult * lr


def create_learning_rate_fn(hyperparameters: spec.Hyperparamters, num_examples):
  steps_per_epoch = num_examples // get_batch_size('imagenet')
  base_learning_rate = (hyperparameters.learning_rate *
                        get_batch_size('imagenet') / 256.)

  def step_fn(step):
    epoch = step / steps_per_epoch
    lr = cosine_decay(base_learning_rate,
                      epoch - hyperparameters.warmup_epochs,
                      hyperparameters.num_epochs - hyperparameters.warmup_epochs)
    warmup = jnp.minimum(1., epoch / hyperparameters.warmup_epochs)
    return lr * warmup
  return step_fn


def optimizer(hyperparameters):
  opt_init_fn, opt_update_fn = optax.chain(
      optax.scale_by_adam(
          b1=1.0 - hyperparameters.one_minus_beta_1,
          b2=0.999,
          eps=hyperparameters.epsilon),
      optax.scale(-hyperparameters.learning_rate)
  )
  return opt_init_fn, opt_update_fn


def init_optimizer_state(
    workload: spec.Workload,
    model_params: spec.ParameterTree,
    model_state: spec.ModelAuxillaryState,
    hyperparameters: spec.Hyperparamters,
    rng: spec.RandomState) -> spec.OptimizerState:
  workload.learning_rate_fn = create_learning_rate_fn(
      hyperparameters,
      workload.num_train_examples)
  params_zeros_like = jax.tree_map(
      lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes)
  opt_init_fn, _ = optimizer(hyperparameters)
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

  lr = workload.learning_rate_fn(step)
  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  aux, grad = grad_fn(current_params)
  _, opt_update_fn = optimizer(hyperparameters)
  new_model_state, logits = aux[1]
  updates, new_optimizer_state = opt_update_fn(
      grad, optimizer_state, current_params)
  updated_params = optax.apply_updates(current_params, updates)
  metrics = workload.compute_metrics(logits, batch['label'])
  metrics['learning_rate'] = lr

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
    model_state = workload.sync_batch_stats(model_state)

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
