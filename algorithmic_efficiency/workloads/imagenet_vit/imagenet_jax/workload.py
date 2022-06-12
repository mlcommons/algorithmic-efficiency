"""ImageNet workload implemented in Jax."""
from typing import Dict, Tuple

from flax import jax_utils
import jax

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax.workload import ImagenetResNetWorkload
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_jax import models
from algorithmic_efficiency.workloads.imagenet_vit.workload import \
    BaseImagenetVitWorkload


# Make sure we inherit from the ViT base workload first.
class ImagenetVitWorkload(BaseImagenetVitWorkload, ImagenetResNetWorkload):

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    model_kwargs = models.decode_variant('S/16')
    model = models.ViT(num_classes=1000, **model_kwargs)
    self._model = model
    params, model_state = self.initialized(rng, model)
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
    del model_state
    del update_batch_norm
    logits = self._model.apply(
        {'params': params},
        augmented_and_preprocessed_input_batch['inputs'],
        mutable=False)
    return logits, None
