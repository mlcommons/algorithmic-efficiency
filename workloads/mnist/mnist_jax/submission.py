"""Training algorithm track submission functions for MNIST."""
from typing import Iterator, List, Tuple, Union

import jax
import jax.numpy as jnp
import optax

import spec
from . import workload


def get_batch_size(workload_name):
  batch_sizes = {'mnist_jax': 1024}
  return batch_sizes[workload_name]


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
  del model_params
  del model_state
  del rng
  params_zeros_like = jax.tree_map(
      lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes)
  opt_init_fn, _ = optimizer(hyperparameters)
  return opt_init_fn(params_zeros_like)


def update_params(
    workload: spec.Workload,
    current_params: spec.ParameterTree,
    current_params_types: spec.ParameterTypeTree,
    model_state: spec.ModelAuxillaryState,
    hyperparameters: spec.Hyperparamters,
    augmented_and_preprocessed_input_batch: spec.Tensor,
    label_batch: spec.Tensor,
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

  def loss_fn(params):
    logits_batch, new_model_state = workload.model_fn(
        params,
        augmented_and_preprocessed_input_batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True)
    loss = workload.loss_fn(label_batch, logits_batch, loss_type)
    return jnp.mean(loss), new_model_state

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, new_model_state), grad = grad_fn(current_params)
  _, opt_update_fn = optimizer(hyperparameters)
  updates, new_optimizer_state = opt_update_fn(
      grad, optimizer_state, current_params)
  updated_params = optax.apply_updates(current_params, updates)
  return new_optimizer_state, updated_params, new_model_state


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
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

  We left out `current_params_types` because we do not believe that it would
  # be necessary for this function.

  Return a tuple of input label batches.
  """
  del optimizer_state
  del current_params
  del global_step
  del rng
  x = next(input_queue)
  return x['image'], x['label']
