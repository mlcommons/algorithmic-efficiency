import functools
import math
from typing import Iterator, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import spec
from flax import jax_utils
from flax import linen as nn
from jax import lax
from workloads.librispeech.librispeech_jax.models import get_seq_lens


def get_batch_size(workload_name):
  batch_sizes = {"librispeech_jax": 8}
  return batch_sizes[workload_name]

def optimizer(hyperparameters: spec.Hyperparamters):
  opt_init_fn, opt_update_fn = optax.adam(
      learning_rate=hyperparameters.learning_rate
    )
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
  optimizer_state = opt_init_fn(params_zeros_like)
  return jax_utils.replicate(optimizer_state), opt_update_fn


# We need to jax.pmap here instead of inside update_params because the latter
# would recompile the function every step.
@functools.partial(
  jax.pmap,
  axis_name='batch',
  in_axes=(None, None, None, None, 0, 0, 0, None, 0, None),
  static_broadcasted_argnums=(0, 1, 2, 3))
def pmapped_train_step(workload, opt_update_fn, in_len, out_len, model_state, optimizer_state,
                       current_param_container, hyperparameters, batch, rng):
  _ = in_len
  _ = out_len
  def _loss_fn(params):
    """loss function used for training."""
    variables = {'params': params, **model_state}
    logits, new_model_state = workload.model_fn(
        params,
        batch['input'],
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True)
    loss = workload.loss_fn(batch['label'], logits)
    return loss, (new_model_state, logits)

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  aux, grad = grad_fn(current_param_container)
  grad = lax.pmean(grad, axis_name='batch')
  new_model_state, logits = aux[1]
  updates, new_optimizer_state = opt_update_fn(
      grad, optimizer_state, current_param_container)
  updated_params = optax.apply_updates(current_param_container, updates)

  return new_model_state, new_optimizer_state, updated_params

def update_params(
    workload: spec.Workload,
    current_param_container: spec.ParameterContainer,
    current_params_types: spec.ParameterTypeTree,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparamters,
    input_batch: spec.Tensor,
    label_batch: spec.Tensor,
    loss_type: spec.LossType,
    optimizer_state: spec.OptimizerState,
    eval_results: List[Tuple[int, float]],
    global_step: int,
    rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  _, features, transcripts, input_lengths,  transcripts_padding, in_len, out_len = input_batch
  features = jnp.expand_dims(features.transpose(0, 2, 1), axis=1)
  num_devices = jax.local_device_count()
  reshaped_features = jnp.reshape(
      features,
      (num_devices, features.shape[0] // num_devices, *features.shape[1:]))
  reshaped_input_lengths = jnp.reshape(
      input_lengths,
      (num_devices, input_lengths.shape[0] // num_devices, *input_lengths.shape[1:]))
  reshaped_transcripts = jnp.reshape(
      transcripts,
      (num_devices, transcripts.shape[0] // num_devices,
       *transcripts.shape[1:]))
  reshaped_transcripts_padding = jnp.reshape(
      transcripts_padding,
      (num_devices, transcripts_padding.shape[0] // num_devices,
       *transcripts_padding.shape[1:]))
  sequential = [
      nn.Conv(
        features=32,
        kernel_size=(41, 11),
        strides=(2, 2),
        padding=((20, 20), (5, 5)),
      ),
      nn.Conv(
        features=32,
        kernel_size=(21, 11),
        strides=(2, 1),
        padding=((10, 10), (5, 5)),
      )
    ]
  output_lengths = get_seq_lens(input_lengths, sequential)
  output_padding = np.zeros((features.shape[0],math.ceil(features.shape[-1]/2)))
  for i,x in enumerate(output_lengths):
    output_padding[i,x:] = 1
  reshaped_output_padding = jnp.reshape(
    output_padding,
      (num_devices, output_padding.shape[0] // num_devices, *output_padding.shape[1:])
  )
  batch = {
    'input': (reshaped_features, reshaped_input_lengths),
    'label': (reshaped_transcripts, reshaped_transcripts_padding, reshaped_output_padding)
  }
  optimizer_state, opt_update_fn = optimizer_state
  new_model_state, new_optimizer_state, new_params = pmapped_train_step(
    workload, opt_update_fn, in_len, out_len, model_state, optimizer_state,
    current_param_container, hyperparameters, batch, rng)

  steps_per_epoch = workload.num_train_examples // get_batch_size('librispeech_jax')
  # due to the psum, batchnorm states don't have to be synchronised
  #if (global_step + 1) % steps_per_epoch == 0:
    #new_model_state = workload.sync_batch_stats(new_model_state)

  return (new_optimizer_state, opt_update_fn), new_params, new_model_state


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
  return next(input_queue), None
