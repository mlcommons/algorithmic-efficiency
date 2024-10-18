import functools
from typing import Dict, Optional, Tuple

from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax.workload import \
    LibriSpeechConformerWorkload
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_jax import \
    models


class LibriSpeechDeepSpeechWorkload(LibriSpeechConformerWorkload):

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
        input_dropout_rate=aux_dropout_rate,
        use_tanh=self.use_tanh,
        enable_residual_connections=self.enable_residual_connections,
        enable_decoder_layer_norm=self.enable_decoder_layer_norm,
        layernorm_everywhere=self.layernorm_everywhere,
        freq_mask_count=self.freq_mask_count,
        time_mask_count=self.time_mask_count,
    )
    self._model = models.Deepspeech(model_config)
    input_shape = [(320000,), (320000,)]
    fake_input_batch = [np.zeros((2, *x), jnp.float32) for x in input_shape]

    model_init_fn = jax.jit(functools.partial(self._model.init, train=False))

    params_rng, dropout_rng = jax.random.split(rng, 2)
    variables = model_init_fn({'params': params_rng, 'dropout': dropout_rng},
                              *fake_input_batch)

    model_state = variables[
        'batch_stats'] if not self.layernorm_everywhere else {}
    params = variables['params']
    self._param_shapes = param_utils.jax_param_shapes(params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
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
      update_batch_norm: bool,
      use_running_average_bn: Optional[bool] = None
  ) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    variables = {'params': params, **model_state}
    inputs, input_paddings = augmented_and_preprocessed_input_batch['inputs']
    is_train_mode = mode == spec.ForwardPassMode.TRAIN
    if update_batch_norm or is_train_mode:
      (logits, logit_paddings), new_model_state = self._model.apply(
          variables,
          inputs,
          input_paddings,
          train=True,
          rngs={'dropout' : rng},
          mutable=['batch_stats'])
      return (logits, logit_paddings), new_model_state
    else:
      logits, logit_paddings = self._model.apply(
          variables,
          inputs,
          input_paddings,
          train=False,
          mutable=False)
      return (logits, logit_paddings), model_state

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key == 'Dense_0'

  @property
  def validation_target_value(self) -> float:
    return 0.119936

  @property
  def test_target_value(self) -> float:
    return 0.074143

  @property
  def step_hint(self) -> int:
    """Max num steps the baseline algo was given to reach the target."""
    return 48_000

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 55_506  # ~15.4 hours

  @property
  def use_tanh(self) -> bool:
    return False

  @property
  def enable_residual_connections(self) -> bool:
    return True

  @property
  def enable_decoder_layer_norm(self) -> bool:
    return True

  @property
  def layernorm_everywhere(self) -> bool:
    return False

  @property
  def freq_mask_count(self) -> int:
    return 2

  @property
  def time_mask_count(self) -> int:
    return 10


class LibriSpeechDeepSpeechTanhWorkload(LibriSpeechDeepSpeechWorkload):

  @property
  def use_tanh(self) -> bool:
    return True

  @property
  def validation_target_value(self) -> float:
    return 0.150883

  @property
  def test_target_value(self) -> float:
    return 0.098613


class LibriSpeechDeepSpeechNoResNetWorkload(LibriSpeechDeepSpeechWorkload):

  @property
  def enable_residual_connections(self) -> bool:
    return False

  @property
  def validation_target_value(self) -> float:
    return 0.131564

  @property
  def test_target_value(self) -> float:
    return 0.079297


class LibriSpeechDeepSpeechNormAndSpecAugWorkload(LibriSpeechDeepSpeechWorkload
                                                 ):

  @property
  def eval_batch_size(self) -> int:
    return 128

  @property
  def enable_decoder_layer_norm(self) -> bool:
    return False

  @property
  def layernorm_everywhere(self) -> bool:
    return True

  @property
  def freq_mask_count(self) -> int:
    return 4

  @property
  def time_mask_count(self) -> int:
    return 15

  @property
  def validation_target_value(self) -> float:
    return 0.14342

  @property
  def test_target_value(self) -> float:
    return 0.090976
