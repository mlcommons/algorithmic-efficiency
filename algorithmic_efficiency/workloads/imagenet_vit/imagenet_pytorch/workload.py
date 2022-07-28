"""ImageNet ViT workload implemented in PyTorch."""

import contextlib
from typing import Dict, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_pytorch.workload import \
    ImagenetResNetWorkload
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_pytorch import \
    models
from algorithmic_efficiency.workloads.imagenet_vit.workload import \
    BaseImagenetVitWorkload
from algorithmic_efficiency.workloads.imagenet_vit.workload import \
    decode_variant

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


# Make sure we inherit from the ViT base workload first.
class ImagenetVitWorkload(BaseImagenetVitWorkload, ImagenetResNetWorkload):

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model_kwargs = decode_variant('S/16')
    model = models.ViT(num_classes=1000, **model_kwargs)
    self._param_shapes = {
        k: spec.ShapeTuple(v.shape) for k, v in model.named_parameters()
    }
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = torch.nn.DataParallel(model)
    return model, None

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del rng

    model = params

    if mode == spec.ForwardPassMode.EVAL:
      if update_batch_norm:
        raise ValueError(
            'Batch norm statistics cannot be updated during evaluation.')
      model.eval()

    if mode == spec.ForwardPassMode.TRAIN:
      model.train()

    contexts = {
        spec.ForwardPassMode.EVAL: torch.no_grad,
        spec.ForwardPassMode.TRAIN: contextlib.nullcontext
    }

    with contexts[mode]():
      logits_batch = model(augmented_and_preprocessed_input_batch['inputs'])

    return logits_batch, None
