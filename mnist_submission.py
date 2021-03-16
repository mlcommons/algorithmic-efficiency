"""Training algorithm track submission functions."""
from typing import Iterator, List, Tuple, Union

import jax
import jax.numpy as jnp
import optax

import mnist_spec
import spec


def _optimizer(hyperparameters):
  opt_init, opt_update = optax.chain(
      optax.scale_by_adam(
          b1=hyperparameters.beta_1,
          b2=hyperparameters.beta_2,
          eps=hyperparameters.epsilon),
      optax.scale(-hyperparameters.learning_rate)
  )
  return opt_init, opt_update


def init_optimizer_state(
    params_shapes: spec.ParameterShapeTree,
    hyperparameters: spec.Hyperparamters,
    rng: spec.RandomState) -> spec.OptimizerState:
  del rng
  opt_init, _ = _optimizer(hyperparameters)
  return opt_init(jax.tree_map(jnp.zeros, params_shapes))


_UpdateReturn = Tuple[
    spec.OptimizerState, spec.ParameterTree, spec.ModelAuxillaryState]


def update_params(
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
    rng: spec.RandomState) -> _UpdateReturn:
  """Return (updated_optimizer_state, updated_params)."""
  del current_params_types
  del eval_results
  del global_step

  def loss_fn():
    logits_batch, new_model_state = mnist_spec.model_fn(
        current_params,
        augmented_and_preprocessed_input_batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True)
    loss = mnist_spec.loss_fn(label_batch, logits_batch, loss_type)
    return loss, new_model_state

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, new_model_state), grad = grad_fn()
  _, opt_update = _optimizer(hyperparameters)
  updates, new_opt_state = opt_update(grad, optimizer_state, current_params)
  updated_params = optax.apply_updates(current_params, updates)
  return new_opt_state, updated_params, new_model_state


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(
    input_queue: Iterator[Tuple[spec.Tensor, spec.Tensor]],
    optimizer_state: spec.OptimizerState,
    current_params: spec.ParameterTree,
    loss_type: spec.LossType,
    hyperparameters: spec.Hyperparamters,
    global_step: int,
    rng: spec.RandomState) -> Tuple[spec.Tensor, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a single training example and label.

  We left out `current_params_types` because we do not believe that it would
  # be necessary for this function.
  """
  del optimizer_state
  del current_params
  del loss_type
  del global_step
  del rng
  return [next(input_queue) for _ in range(hyperparameters.batch_size)]


