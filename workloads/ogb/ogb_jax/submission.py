from typing import Iterator, List, Tuple

import functools
import jax
import jax.numpy as jnp
import optax

import spec


def get_batch_size(workload_name):
  del workload_name
  return 256


def optimizer(hyperparameters: spec.Hyperparamters) -> optax.GradientTransformation:
  """Creates an optimizer."""
  opt_init_fn, opt_update_fn = optax.adam(
      learning_rate=hyperparameters.learning_rate)
  return opt_init_fn, opt_update_fn


def init_optimizer_state(
    workload: spec.Workload,
    model_params: spec.ParameterContainer,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparamters,
    rng: spec.RandomState) -> spec.OptimizerState:
  params_zeros_like = jax.tree_map(
      lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes)
  opt_init_fn, opt_update_fn = optimizer(hyperparameters)
  init_optimizer_state = opt_init_fn(params_zeros_like)
  return init_optimizer_state, opt_update_fn


# We need to jax.pmap here instead of inside update_params because the latter
# would recompile the function every step.
@functools.partial(
    jax.jit,
    static_argnums=(0, 1))
def pmapped_train_step(workload, opt_update_fn, model_state, optimizer_state,
                       current_param_container, hyperparameters, input_batch, label_batch, rng):
  def loss_fn(params):
    logits_batch, new_model_state  = workload.model_fn(
        params,
        input_batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True)
    loss = workload.loss_fn(label_batch, logits_batch)
    return loss, new_model_state

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, new_model_state), grad = grad_fn(current_param_container)
  optimizer_state, opt_update_fn = optimizer_state
  updates, new_optimizer_state = opt_update_fn(
      grad, optimizer_state, current_param_container)
  updated_params = optax.apply_updates(current_param_container, updates)
  return new_model_state, (new_optimizer_state, opt_update_fn), updated_params

def update_params(
    workload: spec.Workload,
    current_param_container: spec.ParameterContainer,
    current_params_types: spec.ParameterTypeTree,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparamters,
    input_batch: spec.Tensor,
    label_batch: spec.Tensor,
    loss_type: spec.LossType,
    # This will define the output activation via `output_activation_fn`.
    optimizer_state: spec.OptimizerState,
    eval_results: List[Tuple[int, float]],
    global_step: int,
    rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del eval_results
  del global_step
  print('\n'*10, current_param_container)

  optimizer_state, opt_update_fn = optimizer_state
  new_model_state, new_optimizer_state, new_params = pmapped_train_step(
      workload, opt_update_fn, model_state, optimizer_state,
      current_param_container, hyperparameters, input_batch, label_batch, rng)

  steps_per_epoch = workload.num_train_examples // get_batch_size('imagenet')
  if (global_step + 1) % steps_per_epoch == 0:
    # sync batch statistics across replicas once per epoch
    new_model_state = workload.sync_batch_stats(new_model_state)

  return new_model_state, new_optimizer_state, new_params

def data_selection(
    workload: spec.Workload,
    input_queue: Iterator[Tuple[spec.Tensor, spec.Tensor]],
    optimizer_state: spec.OptimizerState,
    current_param_container: spec.ParameterContainer,
    hyperparameters: spec.Hyperparamters,
    global_step: int,
    rng: spec.RandomState) -> Tuple[spec.Tensor, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.
  Each element of the queue is a single training example and label.
  Return a tuple of input label batches.
  """
  graphs = next(input_queue)
  labels = graphs.globals
  return graphs, labels
