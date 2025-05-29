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
from algoperf.workloads.lm.input_pipeline import get_hf_dataloader, get_lm_dataset


class LmWorkload(BaseLmWorkload):
  """LM JAX workload."""
  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         num_batches: Optional[int] = None,
                         repeat_final_dataset: bool = False):
    """Build an input queue using pre-cached FineWeb dataset."""
    del num_batches
    del repeat_final_dataset
    loader = get_lm_dataset(
        data_rng=data_rng,
        split=split,
        data_dir=data_dir,
        global_batch_size=global_batch_size)
    return loader

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    
    model = LinearModel(vocab_size=self._vocab_size)
    input_shape = (1, self._seq_len, self._vocab_size)
    params_rng, init_rng = jax.random.split(rng)
    variables = jax.jit(model.init)({'params': params_rng},
                                  jnp.ones(input_shape, jnp.float32))
    params = variables['params'] 
    self._param_shapes = param_utils.jax_param_shapes(params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
    params = sharding_utils.shard_replicated(params)
    model_state = None
    self._model = model
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
    inputs = jax.nn.one_hot(batch['inputs'], self._vocab_size, axis=-1)
    logits = self._model.apply({'params': params}, inputs)
    return logits, None

  def loss_fn(
      self,
      label_batch: spec.Tensor,  # One-hot labels.
      logits_batch: spec.Tensor, # Dense logits.
      mask_batch: Optional[spec.Tensor] = None,
      label_smoothing: Optional[float] = 0.0) -> Dict[str, spec.Tensor]: 
    del mask_batch, label_smoothing
    logits_flat = logits_batch.reshape(-1, self._vocab_size)
    targets = jax.nn.one_hot(label_batch, self._vocab_size, axis=-1)
    targets_flat = targets.reshape(-1, self._vocab_size)
    # Cross-entropy loss
    loss = -jnp.sum(targets_flat * jax.nn.log_softmax(logits_flat, axis=-1))
    n_valid_examples = logits_flat.shape[0]
    return {'summed': loss, 'n_valid_examples': n_valid_examples}

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
