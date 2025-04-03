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
from algoperf.workloads.lm.lm_jax.models import LinearModel
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
    
    self._model = LinearModel(vocab_size=self._vocab_size)
    input_shape = (1, self._seq_len, self._vocab_size)
    params_rng, init_rng = jax.random.split(rng)
    print(params_rng)
    # variables = model.init(init_rng, jnp.ones(input_shape, jnp.float32))
    variables = jax.jit(self._model.init)({'params': params_rng}, jnp.ones(input_shape, jnp.float32))
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
    logits = self._model.apply({'params': params}, inputs)
    return logits, None

  def loss_fn(
      self,
      label_batch: spec.Tensor,
      logits_batch: spec.Tensor,
      mask_batch: Optional[spec.Tensor] = None,
      label_smoothing: float = 0.0) -> Dict[str, spec.Tensor]:
    """Compute cross-entropy loss for language modeling in JAX."""
    vocab_size = logits_batch.shape[-1]
    
    if len(label_batch.shape) == len(logits_batch.shape):
      # One-hot labels
      loss = -jnp.sum(label_batch * jax.nn.log_softmax(logits_batch, axis=-1))
    else:
      # Dense labels
      loss = -jax.nn.log_softmax(logits_batch)[jnp.arange(label_batch.shape[0]), label_batch]
    
    if mask_batch is not None:
      loss = loss * mask_batch
    
    n_valid = mask_batch.sum() if mask_batch is not None else label_batch.shape[0]
    return {
        'summed': loss.sum(),
        'n_valid_examples': n_valid,
        'per_example': loss
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
