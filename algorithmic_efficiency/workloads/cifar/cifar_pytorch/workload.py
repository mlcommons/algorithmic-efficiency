"""CIFAR10 workload implemented in PyTorch."""

import contextlib
import math
import random
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.datasets import CIFAR10

from algorithmic_efficiency import data_utils
from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
import algorithmic_efficiency.random_utils as prng
from algorithmic_efficiency.workloads.cifar.workload import BaseCifarWorkload
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_pytorch.models import \
    resnet18

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


class CifarWorkload(BaseCifarWorkload):

  def _build_input_queue(self,
                         data_rng: spec.RandomState,
                         split: str,
                         data_dir: str,
                         global_batch_size: int):
    it = self._build_dataset(data_rng, split, data_dir, global_batch_size)
    for batch in it:
      yield {
          'inputs': batch['inputs'].to(DEVICE, non_blocking=True),
          'targets': batch['targets'].to(DEVICE, non_blocking=True),
      }

  def _build_dataset(self,
                     data_rng: spec.RandomState,
                     split: str,
                     data_dir: str,
                     batch_size: int):
    is_train = split == 'train'

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[i / 255 for i in self.train_mean],
            std=[i / 255 for i in self.train_stddev])
    ])
    eval_transform_config = normalize
    train_transform_config = transforms.Compose([
        transforms.RandomResizedCrop(
            self.center_crop_size,
            scale=self.scale_ratio_range,
            ratio=self.aspect_ratio_range),
        transforms.RandomHorizontalFlip(),
        normalize
    ])

    transform = train_transform_config if is_train else eval_transform_config
    dataset = CIFAR10(
        root=data_dir,
        train=split in ['train', 'eval_train', 'validation'],
        download=True,
        transform=transform)
    assert self.num_train_examples + self.num_validation_examples == 50000
    indices = list(range(50000))
    random.Random(data_rng[0]).shuffle(indices)
    indices_split = {
        'train': indices[:self.num_train_examples],
        'validation': indices[self.num_train_examples:],
        'eval_train': indices[:self.num_eval_train_examples]
    }
    if split in indices_split:
      dataset = torch.utils.data.Subset(dataset, indices_split[split])

    sampler = None
    if USE_PYTORCH_DDP:
      if is_train:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=N_GPUS, rank=RANK, shuffle=True)
      else:
        sampler = data_utils.DistributedEvalSampler(
            dataset, num_replicas=N_GPUS, rank=RANK, shuffle=False)
      batch_size //= N_GPUS
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not USE_PYTORCH_DDP and is_train,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=is_train)
    dataloader = data_utils.cycle(dataloader, custom_sampler=USE_PYTORCH_DDP)

    return dataloader

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key in ['fc.weight', 'fc.bias']

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model = resnet18(num_classes=10)
    self._param_shapes = param_utils.pytorch_param_shapes(model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = torch.nn.DataParallel(model)
    return model, None

  def _update_batch_norm(self, model, update_batch_norm):
    for m in model.modules():
      if isinstance(
          m,
          (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
        if not update_batch_norm:
          m.eval()
        m.requires_grad_(update_batch_norm)
        m.track_running_stats = update_batch_norm

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
    """Dropout is unused."""
    del model_state
    del rng
    del dropout_rate
    del aux_dropout_rate

    model = params

    if mode == spec.ForwardPassMode.EVAL:
      if update_batch_norm:
        raise ValueError(
            'Batch norm statistics cannot be updated during evaluation.')
      model.eval()

    if mode == spec.ForwardPassMode.TRAIN:
      model.train()
      self._update_batch_norm(model, update_batch_norm)

    contexts = {
        spec.ForwardPassMode.EVAL: torch.no_grad,
        spec.ForwardPassMode.TRAIN: contextlib.nullcontext
    }

    with contexts[mode]():
      logits_batch = model(augmented_and_preprocessed_input_batch['inputs'])

    return logits_batch, None

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(self,
              label_batch: spec.Tensor,
              logits_batch: spec.Tensor,
              mask_batch: Optional[spec.Tensor] = None,
              label_smoothing: float = 0.0) -> spec.Tensor:  # differentiable
    losses = F.cross_entropy(
        logits_batch,
        label_batch,
        reduction='none',
        label_smoothing=label_smoothing)
    # mask_batch is assumed to be shape [batch].
    if mask_batch is not None:
      losses *= mask_batch
    return losses

  def _eval_metric(self, logits, labels):
    """Return the mean accuracy and loss as a dict."""
    predicted = torch.argmax(logits, 1)
    # not accuracy, but nr. of correct predictions
    accuracy = (predicted == labels).sum()
    loss = self.loss_fn(labels, logits).sum()
    return {'accuracy': accuracy, 'loss': loss}

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str,
                           global_step: int = 0):
    """Run a full evaluation of the model."""
    data_rng, model_rng = prng.split(rng, 2)
    if split not in self._eval_iters:
      # These iterators repeat indefinitely.
      self._eval_iters[split] = self._build_input_queue(
          data_rng, split, data_dir, global_batch_size=global_batch_size)

    total_metrics = {
        'accuracy': torch.tensor(0., device=DEVICE),
        'loss': torch.tensor(0., device=DEVICE),
    }
    num_batches = int(math.ceil(num_examples / global_batch_size))
    for _ in range(num_batches):
      batch = next(self._eval_iters[split])
      logits, _ = self.model_fn(
          params,
          batch,
          model_state,
          spec.ForwardPassMode.EVAL,
          model_rng,
          dropout_rate=None,
          aux_dropout_rate=None,
          update_batch_norm=False)
      batch_metrics = self._eval_metric(logits, batch['targets'])
      total_metrics = {
          k: v + batch_metrics[k] for k, v in total_metrics.items()
      }
    if USE_PYTORCH_DDP:
      for metric in total_metrics.values():
        dist.all_reduce(metric)
    return {k: float(v.item() / num_examples) for k, v in total_metrics.items()}
