"""LM workload implemented in Jax."""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils
from algoperf import param_utils
from algoperf import sharding_utils
from algoperf import spec
from algoperf.workloads.lm.workload import BaseLmWorkload
from algoperf.workloads.lm.lm_jax.nanodo_model import (
    TransformerDo, DoConfig, init_rope, apply_rope)
from algoperf.workloads.lm.input_pipeline import get_hf_dataloader


class LmWorkload(BaseLmWorkload):
  """LM JAX workload."""

  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         num_batches: Optional[int] = None,
                         repeat_final_dataset: bool = False):
    """Build an input queue using HuggingFace FineWeb dataset."""
    del num_batches
    del repeat_final_dataset
    loader = get_hf_dataloader(
        cache_dir=data_dir,
        data_rng=data_rng,
        batch_size=global_batch_size,
        seq_len=self._seq_len,
        framework="jax",
        split=split)
    return loader

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    
    # Initialize NanoDO transformer model
    cfg = DoConfig(
        D=512,  # model dim
        H=8,    # num heads
        L=self._seq_len,
        N=6,    # num layers
        V=self._vocab_size,
        F=2048, # feedforward dim
        dtype=jnp.float32
    )
    self._model = TransformerDo(cfg)
    input_shape = (1, self._seq_len)  # For token IDs
    
    params_rng, init_rng = jax.random.split(rng)
    variables = jax.jit(self._model.init)({'params': params_rng}, 
                                        jnp.ones(input_shape, jnp.int32))
    params = variables['params']
    self._param_shapes = param_utils.jax_param_shapes(params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
    params = sharding_utils.shard_replicated(params)
    model_state = None
    return params, model_state

  def model_fn(
      self,
      params: spec.ParameterContainer,
      batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    
    del mode, rng, update_batch_norm, model_state
    inputs = batch['inputs']
    
    # Convert one-hot inputs to token IDs if needed
    if inputs.ndim == 3:  # one-hot encoded
      inputs = jnp.argmax(inputs, axis=-1)
    
    logits = self._model.apply({'params': params}, inputs)
    return logits, None

  def loss_fn(
      self,
      label_batch: spec.Tensor,
      logits_batch: spec.Tensor,
      mask_batch: Optional[spec.Tensor] = None,
      label_smoothing: float = 0.0) -> Dict[str, spec.Tensor]:
    """Compute cross-entropy loss for language modeling in JAX."""
    # Convert one-hot labels to token IDs if needed
    if len(label_batch.shape) == len(logits_batch.shape):  # one-hot
      label_batch = jnp.argmax(label_batch, axis=-1)
    
    # Reshape for sequence modeling
    logits = logits_batch.reshape(-1, logits_batch.shape[-1])
    labels = label_batch.reshape(-1)
    
    # Compute cross-entropy loss
    loss = -jnp.sum(
        jax.nn.log_softmax(logits)[jnp.arange(labels.shape[0]), labels])
    
    if mask_batch is not None:
      mask = mask_batch.reshape(-1)
      loss = loss * mask
      n_valid = mask.sum()
    else:
      n_valid = labels.shape[0]
    
    return {
        'summed': loss,
        'n_valid_examples': n_valid,
        'per_example': loss / n_valid  # Return per-token loss
    }

  def is_output_params(self, param_name: str) -> bool:
    """Return whether the given parameter is an output parameter."""
    return param_name.contains('output') 
    
  def _eval_batch(self,
                  params: spec.ParameterContainer,
                  batch: Dict[str, spec.Tensor],
                  model_state: spec.ModelAuxiliaryState,
                  rng: spec.RandomState) -> spec.Tensor:
    """Evaluate the model on a single batch."""
    logits, _ = self.model_fn(
        params, batch, model_state, spec.ForwardPassMode.EVAL, rng, False)
    targets = batch['targets']
    
    # Calculate cross-entropy loss
    loss = -jnp.sum(targets * jax.nn.log_softmax(logits, axis=-1))
    return loss
