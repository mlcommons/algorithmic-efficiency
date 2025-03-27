"""ImageNet workload implemented in Jax."""

from typing import Dict, Optional, Tuple

from flax import linen as nn
from flax.core import pop
import jax
import jax.numpy as jnp

from algoperf import param_utils
from algoperf import sharding_utils
from algoperf import spec
from algoperf.workloads.imagenet_resnet.imagenet_jax.workload import \
    ImagenetResNetWorkload
from algoperf.workloads.imagenet_vit.imagenet_jax import models
from algoperf.workloads.imagenet_vit.workload import BaseImagenetVitWorkload
from algoperf.workloads.imagenet_vit.workload import decode_variant


# Make sure we inherit from the ViT base workload first.
class ImagenetVitWorkload(BaseImagenetVitWorkload, ImagenetResNetWorkload):

  def initialized(self, key: spec.RandomState,
                  model: nn.Module) -> spec.ModelInitState:
    input_shape = (1, 224, 224, 3)
    params_rng, dropout_rng = jax.random.split(key)
    variables = jax.jit(
        model.init)({'params': params_rng, 'dropout': dropout_rng},
                    jnp.ones(input_shape))
    model_state, params = pop(variables, "params")
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
        use_glu=self.use_glu,
        use_post_layer_norm=self.use_post_layer_norm,
        use_map=self.use_map,
        **decode_variant('S/16'))
    params, model_state = self.initialized(rng, self._model)
    self._param_shapes = param_utils.jax_param_shapes(params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
    params = sharding_utils.shard_replicated(params)
    model_state = sharding_utils.shard_replicated(model_state)
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


class ImagenetVitGluWorkload(ImagenetVitWorkload):

  @property
  def use_glu(self) -> bool:
    return True

  @property
  def validation_target_value(self) -> float:
    return 0.75738

  @property
  def test_target_value(self) -> float:
    return 0.6359


class ImagenetVitPostLNWorkload(ImagenetVitWorkload):

  @property
  def use_post_layer_norm(self) -> bool:
    return True

  @property
  def validation_target_value(self) -> float:
    return 0.75312

  @property
  def test_target_value(self) -> float:
    return 0.6286


class ImagenetVitMapWorkload(ImagenetVitWorkload):

  @property
  def use_map(self) -> bool:
    return True

  @property
  def validation_target_value(self) -> float:
    return 0.77113

  @property
  def test_target_value(self) -> float:
    return 0.6523
