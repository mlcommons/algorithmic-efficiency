"""Training algorithm track submission functions for ImageNet."""

from typing import Iterator, List, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import optax
from flax import jax_utils, optim
import ml_collections

import spec
from . import config as config_lib

config = config_lib.get_config()



def get_batch_size(workload_name):
  del workload_name
  return 128


def cosine_decay(lr, step, total_steps):
  ratio = jnp.maximum(0., step / total_steps)
  mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
  return mult * lr


def create_learning_rate_fn(config: ml_collections.ConfigDict, num_examples):
  steps_per_epoch = (
      num_examples // config.batch_size
  )
  base_learning_rate = config.learning_rate * config.batch_size / 256.

  def step_fn(step):
    epoch = step / steps_per_epoch
    lr = cosine_decay(base_learning_rate,
                      epoch - config.warmup_epochs,
                      config.num_epochs - config.warmup_epochs)
    warmup = jnp.minimum(1., epoch / config.warmup_epochs)
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
  params_zeros_like = jax.tree_map(
      lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes)
  opt_init_fn, _ = optimizer(hyperparameters)
  optimizer_state = opt_init_fn(params_zeros_like)
  optimizer_state = jax_utils.replicate(optimizer_state)
  return optimizer_state


def train_step(apply_fn, model_state, optimizer_state, current_params, step,
    hyperparameters, batch, learning_rate_fn, model_fn, compute_metrics, loss_fn, loss_type):
  def _loss_fn(params):
    """loss function used for training."""
    variables = {'params': params, **model_state}
    logits, new_model_state = model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        update_batch_norm=False,
         mutable=['batch_stats'],
        apply_fn=apply_fn)
    loss = loss_fn(batch['label'], logits, loss_type)
    weight_penalty_params = jax.tree_leaves(variables['params'])
    weight_decay = 0.0001
    weight_l2 = sum([jnp.sum(x ** 2)
                    for x in weight_penalty_params
                    if x.ndim > 1])
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_model_state, logits)

  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  aux, grad = grad_fn(current_params)
  _, opt_update_fn = optimizer(hyperparameters)
  new_model_state, logits = aux[1]
  updates, new_optimizer_state = opt_update_fn(
      grad, optimizer_state, current_params)
  updated_params = optax.apply_updates(current_params, updates)
  metrics = compute_metrics(logits, batch['label'])
  metrics['learning_rate'] = lr

  return new_model_state, new_optimizer_state, updated_params, metrics


def eval_step(apply_fn, state, params, batch, model_fn, compute_metrics):
  logits, _ = model_fn(
      params,
      batch,
      state,
      spec.ForwardPassMode.EVAL,
      update_batch_norm=False,
      mutable=False,
      apply_fn=apply_fn)
  return compute_metrics(logits, batch['label'])


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

  step = jax_utils.replicate(global_step) # TODO REPLACE THIS WITH STATIC ARGS
  hyperparameters = jax_utils.replicate(hyperparameters) # TODO REPLACE THIS WITH STATIC ARGS
  new_model_state, new_optimizer, new_params, metrics = workload.p_train_step(
    model_state, optimizer_state, current_params, step, hyperparameters, batch)

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
