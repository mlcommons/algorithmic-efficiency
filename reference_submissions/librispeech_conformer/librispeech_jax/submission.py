"""Training algorithm track submission functions for LibriSpeech."""
from typing import Dict, Iterator, List, Tuple

import jax.numpy as jnp
import flax 
import optax

from algorithmic_efficiency import spec


def get_batch_size(workload_name):
  # Return the global batch size.
  del workload_name
  return 64


def cosine_decay(lr, step, total_steps):
  ratio = jnp.maximum(0., step / total_steps)
  mult = 0.5 * (1. + jnp.cos(jnp.pi * ratio))
  return mult * lr


def create_learning_rate_fn(hparams: spec.Hyperparameters,
                            steps_per_epoch: int):
  """Create learning rate schedule."""
  base_learning_rate = hparams.learning_rate * get_batch_size('librispeech') / 256.
  warmup_fn = optax.linear_schedule(
      init_value=0.,
      end_value=base_learning_rate,
      transition_steps=hparams.warmup_epochs * steps_per_epoch)
  cosine_epochs = max(hparams.num_epochs - hparams.warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[hparams.warmup_epochs * steps_per_epoch])
  return schedule_fn


def optimizer(hyperparameters: spec.Hyperparameters, num_train_examples: int):
  steps_per_epoch = num_train_examples // get_batch_size('librispeech')
  learning_rate_fn = create_learning_rate_fn(hyperparameters, steps_per_epoch)
  opt_init_fn, opt_update_fn = optax.adamw(
      b1=hyperparameters.beta1,
      b2=hyperparameters.beta2,
      eps=hyperparameters.epsilon,
      learning_rate=learning_rate_fn)
  return opt_init_fn, opt_update_fn


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
  opt_init_fn, opt_update_fn = optimizer(hyperparameters,
                                         workload.num_train_examples)
  optimizer_state = opt_init_fn(params_zeros_like)
  return jax_utils.replicate(optimizer_state), opt_update_fn


def update_params(
    workload: spec.Workload,
    current_param_container: spec.ParameterContainer,
    current_params_types: spec.ParameterTypeTree,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparameters,
    batch: Dict[str, spec.Tensor],
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
  del hyperparameters


  (outputs, output_paddings), _ = workload.model_fn(
      current_param_container, batch, None,
      spec.ForwardPassMode.TRAIN, rng, False)

  train_ctc_loss = torch.mean(workload.loss_fn(batch, (log_y, output_lengths)))
  optimizer_state.step()

  return optimizer_state, current_param_container, None


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a batch of training examples and labels.
  """
  del optimizer_state
  del current_param_container
  del global_step
  del rng
  del hyperparameters
  del workload
  return next(input_queue)
