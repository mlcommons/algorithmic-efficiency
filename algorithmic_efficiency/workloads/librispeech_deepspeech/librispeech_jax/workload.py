import functools
from typing import Optional

from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech_deepspeech.workload import \
    BaseDeepspeechLibrispeechWorkload
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_jax import \
    models


class LibriSpeechDeepSpeechWorkload(BaseDeepspeechLibrispeechWorkload):

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """Deepspeech model init function.

    Here we use dropout_rate as feed_forward_dropout_rate, and aux_dropout_rate
    as input_dropout_rate.
    """
    model_config = models.DeepspeechConfig(
        feed_forward_dropout_rate=dropout_rate,
        use_specaug=self.use_specaug,
        input_dropout_rate=aux_dropout_rate)
    self._model = models.Deepspeech(model_config)
    input_shape = [(320000,), (320000,)]
    fake_input_batch = [np.zeros((2, *x), jnp.float32) for x in input_shape]

    model_init_fn = jax.jit(functools.partial(self._model.init, train=False))

    params_rng, dropout_rng = jax.random.split(rng, 2)
    variables = model_init_fn({'params': params_rng, 'dropout': dropout_rng},
                              *fake_input_batch)

    model_state = variables['batch_stats']
    params = variables['params']
    self._param_shapes = param_utils.jax_param_shapes(params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
    model_state = jax_utils.replicate(model_state)
    params = jax_utils.replicate(params)
    return params, model_state

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key == 'Dense_0'
