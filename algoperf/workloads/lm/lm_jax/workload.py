"""LM workload implemented in Jax."""

from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from flax import jax_utils
from algoperf import param_utils
from algoperf import spec
from algoperf.workloads.lm.workload import BaseLmWorkload
from algoperf.workloads.lm.lm_jax.models import LinearModel


class LmWorkload(BaseLmWorkload):
  """LM JAX workload."""

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    
    model = LinearModel(vocab_size=self._vocab_size)
    input_shape = (1, self._seq_len, self._vocab_size)
    variables = model.init(rng, jnp.ones(input_shape, jnp.float32))
    model_state, params = variables.pop('params')
    
    self._param_shapes = param_utils.jax_param_shapes(params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
    model_state = jax_utils.replicate(model_state)
    params = jax_utils.replicate(params)
    
    return params, model_state

  def model_fn(
      self,
      params: spec.ParameterContainer,
      batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    
    del mode, rng, update_batch_norm  # Not used for linear model
    inputs = batch['inputs']
    logits = self._model.apply({'params': params, **model_state}, inputs)
    return logits, model_state

  def _eval_batch(self,
                  params: spec.ParameterContainer,
                  batch: Dict[str, spec.Tensor],
                  model_state: spec.ModelAuxiliaryState,
                  rng: spec.RandomState) -> spec.Tensor:
    """Evaluate the model on a single batch."""
    pass
