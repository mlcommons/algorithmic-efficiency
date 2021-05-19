"""Training algorithm track submission functions for MNIST."""
from typing import Iterator, List, Tuple

from . import config
from . import models
from . import train
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import spec


def get_batch_size(workload_name):
  batch_sizes = {"wmt_jax": config.config.per_device_batch_size}
  return batch_sizes[workload_name]


def optimizer(hyperparameters):
  opt_init_fn, opt_update_fn = optax.chain(
      optax.scale_by_adam(
          b1=1.0 - hyperparameters.one_minus_beta_1,
          b2=0.98,
          eps=hyperparameters.epsilon),
      optax.scale(-hyperparameters.learning_rate))
  return opt_init_fn, opt_update_fn


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterTree,
                         model_state: spec.ModelAuxillaryState,
                         hyperparameters: spec.Hyperparamters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del model_params
  del model_state
  del rng
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  opt_init_fn, _ = optimizer(hyperparameters)
  return opt_init_fn(params_zeros_like)


def update_params(
    workload: spec.Workload,
    current_params: spec.ParameterTree,
    current_params_types: spec.ParameterTypeTree,
    model_state: spec.ModelAuxillaryState,
    hyperparameters: spec.Hyperparamters,
    augmented_and_preprocessed_input_batch: spec.Tensor,
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
  del model_state
  del loss_type
  del rng

  train_keys = [
      "inputs", "targets", "inputs_position", "targets_position",
      "inputs_segmentation", "targets_segmentation"
  ]
  (inputs, targets, inputs_positions, targets_positions, inputs_segmentation,
   targets_segmentation) = [
       augmented_and_preprocessed_input_batch.get(k, None) for k in train_keys
   ]

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

  def loss_fn(params):
    """loss function used for training."""
    logits = models.Transformer(workload.train_config).apply(
        {"params": params},
        inputs,
        targets,
        inputs_positions=inputs_positions,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation)

    loss, weight_sum = train.compute_weighted_cross_entropy(
        logits, targets, weights, config.config.label_smoothing)
    mean_loss = loss / weight_sum
    return mean_loss, None

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, new_model_state), grad = grad_fn(current_params)
  _, opt_update_fn = optimizer(hyperparameters)
  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_params)
  updated_params = optax.apply_updates(current_params, updates)
  return new_optimizer_state, updated_params, new_model_state


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Tuple[spec.Tensor, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_params: spec.ParameterTree,
                   hyperparameters: spec.Hyperparamters, global_step: int,
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
  del hyperparameters
  del workload

  return common_utils.shard(jax.tree_map(np.asarray, next(input_queue)))

