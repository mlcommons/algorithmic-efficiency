"""Training algorithm track submission functions for MNIST."""

import functools
from typing import Dict, Iterator, List, Tuple, Any

from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec


def get_batch_size(workload_name):
  # Return the global batch size.
  batch_sizes = {'mnist': 1024}
  return batch_sizes[workload_name]


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
  opt_init_fn, opt_update_fn = optax.chain(
      optax.scale_by_adam(
          b1=1.0 - hyperparameters.one_minus_beta_1,
          b2=0.999,
          eps=hyperparameters.epsilon),
      optax.scale(-hyperparameters.learning_rate))
  return jax_utils.replicate(opt_init_fn(params_zeros_like)), opt_update_fn


# We need to jax.pmap here instead of inside update_params because the latter
# would recompile the function every step.
@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, None, 0, 0, None, 0, 0, 0),
    static_broadcasted_argnums=(0, 1))
def pmapped_update_params(workload: spec.Workload,
                          opt_update_fn,
                          current_param_container: spec.ParameterContainer,
                          model_state: spec.ModelAuxiliaryState,
                          hyperparameters: spec.Hyperparameters,
                          batch: Dict[str, spec.Tensor],
                          optimizer_state: spec.OptimizerState,
                          rng: spec.RandomState) -> spec.UpdateReturn:
  del hyperparameters

  def loss_fn(params):
    logits_batch, new_model_state = workload.model_fn(
        params=params,
        augmented_and_preprocessed_input_batch=batch,
        model_state=model_state,
        mode=spec.ForwardPassMode.TRAIN,
        rng=rng,
        update_batch_norm=True)
    loss_dict = workload.loss_fn(batch['targets'], logits_batch)
    loss = loss_dict['summed'] / loss_dict['n_valid_examples']
    return loss, new_model_state

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, new_model_state), grad = grad_fn(current_param_container)
  grad = lax.pmean(grad, axis_name='batch')
  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
  updated_params = optax.apply_updates(current_param_container, updates)
  return new_optimizer_state, updated_params, new_model_state


def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  train_state: Dict[str, Any],
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del train_state
  del eval_results
  del global_step

  per_device_rngs = jax.random.split(rng, jax.local_device_count())
  optimizer_state, opt_update_fn = optimizer_state
  new_optimizer_state, updated_params, new_model_state = pmapped_update_params(
      workload,
      opt_update_fn,
      current_param_container,
      model_state,
      hyperparameters,
      batch,
      optimizer_state,
      per_device_rngs)
  return (new_optimizer_state, opt_update_fn), updated_params, new_model_state


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
