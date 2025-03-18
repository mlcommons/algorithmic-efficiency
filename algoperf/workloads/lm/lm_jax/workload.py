"""LM workload implemented in Jax."""

import functools
from typing import Any, Dict, Iterator, Optional, Tuple

from absl import logging
from flax import jax_utils
from flax import linen as nn
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax

from algoperf import param_utils
from algoperf import pytorch_utils
from algoperf import spec
from algoperf.workloads.lm.workload import BaseLmWorkload

class LmWorkload(BaseLmWorkload):
  """LM JAX workload."""

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """aux_dropout_rate is used as attention_dropout_rate."""
    pass

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    pass

  def _eval_batch(self,
                  params: spec.ParameterContainer,
                  batch: Dict[str, spec.Tensor],
                  model_state: spec.ModelAuxiliaryState,
                  rng: spec.RandomState) -> spec.Tensor:
    """Evaluate the model on a single batch."""
    pass
