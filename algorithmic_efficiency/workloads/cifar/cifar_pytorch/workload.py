"""CIFAR10 workload implemented in PyTorch."""

import contextlib
import functools
import random
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.datasets import CIFAR10

from algorithmic_efficiency import data_utils
from algorithmic_efficiency import param_utils
from algorithmic_efficiency import pytorch_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.cifar.cifar_pytorch.models import \
    resnet18
from algorithmic_efficiency.workloads.cifar.workload import BaseCifarWorkload

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()


class CifarWorkload(BaseCifarWorkload):

  def _build_dataset(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      global_batch_size: int,
      cache: Optional[bool] = None,
      repeat_final_dataset: Optional[bool] = None
  ) -> torch.utils.data.DataLoader:
    del cache
    del repeat_final_dataset
    is_train = split == 'train'
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=self.train_mean, std=self.train_stddev),
    ])
    eval_transform_config = normalize
    train_transform_config = transforms.Compose([
        transforms.RandomCrop(
            size=self.crop_size,
            padding=self.padding_size,
        ),
        transforms.RandomHorizontalFlip(),
        normalize,
    ])

    transform = train_transform_config if is_train else eval_transform_config
    dataset = CIFAR10(
        root=data_dir,
        train=split in ['train', 'eval_train', 'validation'],
        download=True,
        transform=transform)
    assert self.num_train_examples + self.num_validation_examples == 50000
    indices = list(range(50000))
    indices_split = {
        'train': indices[:self.num_train_examples],
        'validation': indices[self.num_train_examples:],
    }
    if split == 'eval_train':
      train_indices = indices_split['train']
      random.Random(data_rng[0]).shuffle(train_indices)
      indices_split['eval_train'] = train_indices[:self.num_eval_train_examples]
    if split in indices_split:
      dataset = torch.utils.data.Subset(dataset, indices_split[split])

    sampler = None
    if USE_PYTORCH_DDP:
      per_device_batch_size = global_batch_size // N_GPUS
      ds_iter_batch_size = per_device_batch_size
      if is_train:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=N_GPUS, rank=RANK, shuffle=True)
      else:
        sampler = data_utils.DistributedEvalSampler(
            dataset, num_replicas=N_GPUS, rank=RANK, shuffle=False)
    else:
      ds_iter_batch_size = global_batch_size

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=ds_iter_batch_size,
        shuffle=not USE_PYTORCH_DDP and is_train,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=is_train)
    dataloader = data_utils.PrefetchedWrapper(dataloader, DEVICE)
    dataloader = data_utils.cycle(dataloader, custom_sampler=USE_PYTORCH_DDP)
    return dataloader

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """Dropout is unused."""
    del dropout_rate
    del aux_dropout_rate
    torch.random.manual_seed(rng[0])
    model = resnet18(num_classes=self._num_classes)
    self._param_shapes = param_utils.pytorch_param_shapes(model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = torch.nn.DataParallel(model)
    return model, None

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key in ['fc.weight', 'fc.bias']

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
      model.apply(
          functools.partial(
              pytorch_utils.update_batch_norm_fn,
              update_batch_norm=update_batch_norm))
    contexts = {
        spec.ForwardPassMode.EVAL: torch.no_grad,
        spec.ForwardPassMode.TRAIN: contextlib.nullcontext,
    }
    with contexts[mode]():
      logits_batch = model(augmented_and_preprocessed_input_batch['inputs'])
    return logits_batch, None

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,  # Dense or one-hot labels.
      logits_batch: spec.Tensor,
      mask_batch: Optional[spec.Tensor] = None,
      label_smoothing: float = 0.0) -> Dict[str, spec.Tensor]:  # differentiable
    """Evaluate the (masked) loss function at (label_batch, logits_batch).

    Return {'summed': scalar summed loss, 'n_valid_examples': scalar number of
    valid examples in batch, 'per_example': 1-d array of per-example losses}
    (not synced across devices).
    """
    per_example_losses = F.cross_entropy(
        logits_batch,
        label_batch,
        reduction='none',
        label_smoothing=label_smoothing)
    # `mask_batch` is assumed to be shape [batch].
    if mask_batch is not None:
      per_example_losses *= mask_batch
      n_valid_examples = mask_batch.sum()
    else:
      n_valid_examples = len(per_example_losses)
    summed_loss = per_example_losses.sum()
    return {
        'summed': summed_loss,
        'n_valid_examples': torch.tensor(n_valid_examples, device=DEVICE),
        'per_example': per_example_losses,
    }

  def _eval_model(
      self,
      params: spec.ParameterContainer,
      batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      rng: spec.RandomState) -> Dict[spec.Tensor, spec.ModelAuxiliaryState]:
    """Return the mean accuracy and loss as a dict."""
    logits, _ = self.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.EVAL,
        rng,
        update_batch_norm=False)
    targets = batch['targets']
    weights = batch.get('weights')
    if weights is None:
      weights = torch.ones(len(logits)).to(DEVICE)
    _, predicted = torch.max(logits.data, 1)
    # Number of correct predictions.
    accuracy = ((predicted == targets) * weights).sum()
    summed_loss = self.loss_fn(targets, logits, weights)['summed']
    return {'accuracy': accuracy, 'loss': summed_loss}

  def _normalize_eval_metrics(
      self, num_examples: int, total_metrics: Dict[str,
                                                   Any]) -> Dict[str, float]:
    if USE_PYTORCH_DDP:
      for metric in total_metrics.values():
        dist.all_reduce(metric)
    return {k: float(v.item() / num_examples) for k, v in total_metrics.items()}
