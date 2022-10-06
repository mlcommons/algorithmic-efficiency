import functools
from typing import Dict, Optional, Tuple

from absl import flags
from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax.workload import \
    LibriSpeechConformerWorkload
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_jax import \
    models
from algorithmic_efficiency.workloads.librispeech_deepspeech.workload import \
    BaseDeepspeechLibrispeechWorkload

FLAGS = flags.FLAGS


class LibriSpeechDeepSpeechWorkload(LibriSpeechConformerWorkload,
                                    BaseDeepspeechLibrispeechWorkload):

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

    model_state = variables['batch_stats']
    params = variables['params']

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
      dropout_rate: Optional[float],
      aux_dropout_rate: Optional[float],
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    """Deepspeech model function.

    Here we use dropout_rate as feed_forward_dropout_rate, and aux_dropout_rate
    as input_dropout_rate.
    """
    model_config = models.DeepspeechConfig(
        feed_forward_dropout_rate=dropout_rate,
        input_dropout_rate=aux_dropout_rate)
    model = models.Deepspeech(model_config)
    return self._model_fn(params,
                          augmented_and_preprocessed_input_batch,
                          model_state,
                          mode,
                          rng,
                          update_batch_norm,
                          model)
