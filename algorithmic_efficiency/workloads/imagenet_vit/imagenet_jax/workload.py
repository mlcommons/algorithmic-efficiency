"""ImageNet workload implemented in Jax."""

from typing import Dict, Optional, Tuple

from flax import jax_utils
from flax import linen as nn
import jax
import jax.numpy as jnp

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax.workload import \
    ImagenetResNetWorkload
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_jax import models
from algorithmic_efficiency.workloads.imagenet_vit.workload import \
    BaseImagenetVitWorkload
from algorithmic_efficiency.workloads.imagenet_vit.workload import \
    decode_variant


# Make sure we inherit from the ViT base workload first.
class ImagenetVitWorkload(BaseImagenetVitWorkload, ImagenetResNetWorkload):

  def initialized(self, key: spec.RandomState,
                  model: nn.Module) -> spec.ModelInitState:
    input_shape = (1, 224, 224, 3)
    variables = jax.jit(model.init)({'params': key}, jnp.ones(input_shape))
    model_state, params = variables.pop('params')
    return params, model_state

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    del aux_dropout_rate
    self._model = models.ViT(
        dropout_rate=dropout_rate,
        num_classes=self._num_classes,
        use_global_avg_pool=self.use_global_avg_pool,
        use_post_layer_norm=self.use_post_layer_norm,
        **decode_variant('S/16'))
    params, model_state = self.initialized(rng, self._model)
    self._param_shapes = param_utils.jax_param_shapes(params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
    model_state = jax_utils.replicate(model_state)
    params = jax_utils.replicate(params)
    return params, model_state

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key == 'head'

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del update_batch_norm
    train = mode == spec.ForwardPassMode.TRAIN
    logits = self._model.apply({'params': params},
                               augmented_and_preprocessed_input_batch['inputs'],
                               rngs={'dropout': rng},
                               train=train)
    return logits, None

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str,
                           global_step: int = 0) -> Dict[str, float]:
    model_state = None
    return super()._eval_model_on_split(split,
                                        num_examples,
                                        global_batch_size,
                                        params,
                                        model_state,
                                        rng,
                                        data_dir,
                                        global_step)


class ImagenetVitMapWorkload(ImagenetVitWorkload):

  @property
  def use_global_avg_pool(self) -> bool:
    return False


class ImagenetVitPostLayerNormWorkload(ImagenetVitWorkload):

  @property
  def use_post_layer_norm(self) -> bool:
    return True

