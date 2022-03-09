"""Training algorithm track submission functions for MNIST."""

import functools
from typing import Iterator, List, Tuple

from flax import jax_utils
import jax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec


def get_batch_size(workload_name):
  # Return the global batch size.
  batch_sizes = {'mnist': 1024}
  return batch_sizes[workload_name]


def optimizer(hyperparameters):
  opt_init_fn, opt_update_fn = optax.chain(
      optax.scale_by_adam(
          b1=1.0 - hyperparameters.one_minus_beta_1,
          b2=0.999,
          eps=hyperparameters.epsilon),
      optax.scale(-hyperparameters.learning_rate))
  return opt_init_fn, opt_update_fn


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparamters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del model_params
  del model_state
  del rng
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  opt_init_fn, _ = optimizer(hyperparameters)
  return opt_init_fn(params_zeros_like)


# We need to jax.pmap here instead of inside update_params because the latter
# the latter would recompile the function every step.
@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, 0, 0, None, 0, 0, 0, None, 0),
    static_broadcasted_argnums=(0,))
def pmapped_update_params(workload: spec.Workload,
                          current_param_container: spec.ParameterContainer,
                          model_state: spec.ModelAuxiliaryState,
                          hyperparameters: spec.Hyperparamters,
                          input_batch: spec.Tensor,
                          label_batch: spec.Tensor,
                          optimizer_state: spec.OptimizerState,
                          rng: spec.RandomState,
                          local_device_index) -> spec.UpdateReturn:
  # Note that `rng` is the same across all devices! If a per-device RNG is
  # required, then `local_device_index` can be folded into `rng`.
  del local_device_index

  def loss_fn(params):
    logits_batch, new_model_state = workload.model_fn(
        params,
        input_batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True)
    loss = workload.loss_fn(label_batch, logits_batch)
    return jnp.mean(loss), new_model_state

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, new_model_state), grad = grad_fn(current_param_container)
  _, opt_update_fn = optimizer(hyperparameters)
  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
  updated_params = optax.apply_updates(current_param_container, updates)
  return new_optimizer_state, updated_params, new_model_state


def update_params(
    workload: spec.Workload,
    current_param_container: spec.ParameterContainer,
    current_params_types: spec.ParameterTypeTree,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparamters,
    input_batch: spec.Tensor,
    label_batch: spec.Tensor,
    # This will define the output activation via `output_activation_fn`.
    loss_type: spec.LossType,
    optimizer_state: spec.OptimizerState,
    eval_results: List[Tuple[int, float]],
    global_step: int,
    rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del eval_results
  del global_step

  num_devices = jax.local_device_count()
  input_shape = input_batch.shape
  reshaped_input_batch = jnp.reshape(
      input_batch,
      (num_devices, input_shape[0] // num_devices, *input_shape[1:]))
  reshaped_label_batch = jnp.reshape(
      label_batch,
      (num_devices, label_batch.shape[0] // num_devices,
       *label_batch.shape[1:]))

  # TODO(znado) we should be more efficient than replicating state each step.
  new_optimizer_state, updated_params, new_model_state = pmapped_update_params(
      workload, jax_utils.replicate(current_param_container),
      jax_utils.replicate(model_state), hyperparameters, reshaped_input_batch,
      reshaped_label_batch, jax_utils.replicate(optimizer_state), rng,
      jnp.arange(num_devices))
  return (jax_utils.unreplicate(new_optimizer_state),
          jax_utils.unreplicate(updated_params),
          jax_utils.unreplicate(new_model_state))


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Tuple[spec.Tensor, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   hyperparameters: spec.Hyperparamters,
                   global_step: int,
                   rng: spec.RandomState) -> Tuple[spec.Tensor, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a single training example and label.

  We left out `current_params_types` because we do not believe that it would
  # be necessary for this function.

  Return a tuple of input label batches.
  """
  del workload
  del optimizer_state
  del current_param_container
  del hyperparameters
  del global_step
  del rng
  return next(input_queue)
