import functools
import itertools
import math
from typing import Dict, Tuple

from absl import flags
from absl import logging
from flax import jax_utils
import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax

from algorithmic_efficiency import spec

from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax.workload import LibriSpeechConformerWorkload
from algorithmic_efficiency.workloads.librispeech_deepspeech.workload import BaseDeepspeechLibrispeechWorkload
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_jax import \
    models

FLAGS = flags.FLAGS


class LibriSpeechDeepSpeechWorkload(LibriSpeechConformerWorkload, BaseDeepspeechLibrispeechWorkload):

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    model_cls = getattr(models, 'Deepspeech')
    model = model_cls(models.DeepspeechConfig())
    self._model = model
    input_shape = [(320000,), (320000,)]
    fake_input_batch = [np.zeros((2, *x), jnp.float32) for x in input_shape]

    model_init_fn = jax.jit(functools.partial(model.init, train=False))

    params_rng, dropout_rng = jax.random.split(rng, 2)
    variables = model_init_fn({'params': params_rng, 'dropout': dropout_rng},
                              *fake_input_batch)

    model_state=variables["batch_stats"]
    params = variables["params"]

    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      params)
    model_state = jax_utils.replicate(model_state)
    params = jax_utils.replicate(params)
    return params, model_state

