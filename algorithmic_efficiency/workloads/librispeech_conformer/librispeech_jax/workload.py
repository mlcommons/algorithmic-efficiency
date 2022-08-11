import itertools
import math
from typing import Dict, Optional, Tuple

from absl import flags
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import jax_utils
import numpy as np
import functools 

from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech_conformer import metrics
from algorithmic_efficiency.workloads.librispeech_conformer import workload
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax import models
from algorithmic_efficiency import param_utils

FLAGS = flags.FLAGS

class LibriSpeechConformerWorkload(workload.BaseLibrispeechWorkload):
    def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
        model_cls = getattr(models, 'Conformer')
        model = model_cls(models.ConformerConfig())
        self._model = model
        input_shape = [(320000,), (320000,)]
        fake_input_batch = [np.zeros((2, *x), jnp.float32) for x in input_shape]

        model_init_fn = jax.jit(functools.partial(model.init, train=False))

        params_rng, dropout_rng = jax.random.split(rng, 2)
        variables = model_init_fn({'params' : params_rng, 'dropout' : dropout_rng}, *fake_input_batch)

        model_state, params = variables.pop('params')

        self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                        params)
        model_state = jax_utils.replicate(model_state)
        params = jax_utils.replicate(params)
        return params, model_state


    def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
        del rng
        variables = {'params': params, **model_state}
        
        train = mode == spec.ForwardPassMode.TRAIN
        if train:
            (logits, logit_paddings), new_model_state = self._model.apply(
                variables,
                augmented_and_preprocessed_input_batch['inputs'],
                augmented_and_preprocessed_input_batch['input_paddings'],
                train,
                mutable=['batch_stats'])
            return (logits, logit_paddings), new_model_state
        else:
            logits = self._model.apply(
                variables,
                augmented_and_preprocessed_input_batch['inputs'],
                augmented_and_preprocessed_input_batch['input_paddings'],
                train,
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