import itertools
import math
from typing import Dict, Optional, Tuple

from absl import flags
import jax
import jax.numpy as jnp
import flax.linen as nn

from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech_conformer import metrics
from algorithmic_efficiency.workloads.librispeech_conformer import workload
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax import models

FLAGS = flags.FLAGS

class LibriSpeechConformerWorkload(workload.BaseLibrispeechWorkload):
    def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
        model_cls = getattr(models, 'Conformer')
        model = model_cls(models.ConformerConfig())
        self._model = model
        input_shape = (320000, 320000)

        variables = jax.jit(model.init)({'params': rng},
                                        jnp.ones(input_shape, model.config.dtype))
        model_state, params = variables.pop('params')

        self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                        params)
        model_state = jax_utils.replicate(model_state)
        params = jax_utils.replicate(params)
        return params, model_state, model_fn, model_params_types


    def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
        del mode
        del rng
        
        variables = {'params': params, **model_state}
        
        if mode == spec.ForwardPassMode.Train:
            (logits, logit_paddings), new_model_state = self._model.apply(
                variables,
                augmented_and_preprocessed_input_batch['inputs'],
                augmented_and_preprocessed_input_batch['input_paddings'],
                mutable=['batch_stats'])
            return (logits, logit_paddings), new_model_state
        else:
            logits = self._model.apply(
                variables,
                augmented_and_preprocessed_input_batch['inputs'],
                augmented_and_preprocessed_input_batch['input_paddings'],
                mutable=False)
            return logits, None

    @property
    def model_params_types(self):
        if self._param_shapes is None:
            raise ValueError(
                'This should not happen, workload.init_model_fn() should be called '
                'before workload.param_shapes!')
        if self._param_types is None:
            self._param_types = param_utils.jax_param_types(
                self._param_shapes.unfreeze())
        return self._param_types