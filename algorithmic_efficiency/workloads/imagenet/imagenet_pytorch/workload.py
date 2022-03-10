"""ImageNet workload implemented in PyTorch."""
import contextlib
import os
from typing import Tuple

import random_utils as prng
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.imagenet.imagenet_pytorch.models import \
    resnet50
from algorithmic_efficiency.workloads.imagenet.workload import ImagenetWorkload

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# from https://github.com/pytorch/pytorch/issues/23900#issuecomment-518858050
def cycle(iterable):
  iterator = iter(iterable)
  while True:
    try:
      yield next(iterator)
    except StopIteration:
      iterator = iter(iterable)


class ImagenetPytorchWorkload(ImagenetWorkload):

  @property
  def param_shapes(self):
    """
    TODO: return shape tuples from model as a tree
    """
    raise NotImplementedError

  def eval_model(self,
                 params: spec.ParameterContainer,
                 model_state: spec.ModelAuxiliaryState,
                 rng: spec.RandomState,
                 data_dir: str):
    """Run a full evaluation of the model."""
    data_rng, model_rng = prng.split(rng, 2)
    eval_batch_size = 128
    if self._eval_ds is None:
      self._eval_ds = self._build_dataset(
          data_rng, 'test', data_dir, batch_size=eval_batch_size)

    total_metrics = {
        'accuracy': 0.,
        'loss': 0.,
    }
    n_data = 0
    for (images, labels) in self._eval_ds:
      images = images.float().to(DEVICE)
      labels = labels.float().to(DEVICE)
      logits, _ = self.model_fn(
          params,
          images,
          model_state,
          spec.ForwardPassMode.EVAL,
          model_rng,
          update_batch_norm=False)
      batch_metrics = self._eval_metric(logits, labels)
      total_metrics = {
          k: v + batch_metrics[k] for k, v in total_metrics.items()
      }
      n_data += batch_metrics['n_data']
    return {k: float(v / n_data) for k, v in total_metrics.items()}

  def _build_dataset(self,
                     data_rng: spec.RandomState,
                     split: str,
                     data_dir: str,
                     batch_size: int):

    is_train = (split == "train")

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[i / 255 for i in self.train_mean],
            std=[i / 255 for i in self.train_stddev])
    ])
    transform_config = {
        "train":
            transforms.Compose([
                transforms.RandomResizedCrop(
                    self.center_crop_size,
                    scale=self.scale_ratio_range,
                    ratio=self.aspect_ratio_range),
                transforms.RandomHorizontalFlip(),
                normalize
            ]),
        "test":
            transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(self.center_crop_size),
                normalize
            ])
    }

    folder = {'train': 'train', 'test': 'val'}

    dataset = ImageFolder(
        os.path.join(data_dir, folder[split]),
        transform=transform_config[split])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=5,
        pin_memory=True,
        drop_last=is_train)

    if is_train:
      dataloader = cycle(dataloader)

    return dataloader

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model = resnet50()
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    model.to(DEVICE)
    return model, None

  def _update_batch_norm(self, model, update_batch_norm):
    for m in model.modules():
      if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if not update_batch_norm:
          m.eval()
        m.requires_grad_(update_batch_norm)
        m.track_running_stats = update_batch_norm

  def model_fn(
      self,
      params: spec.ParameterContainer,
      input_batch: spec.Tensor,
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
      self._update_batch_norm(model, update_batch_norm)

    contexts = {
        spec.ForwardPassMode.EVAL: torch.no_grad,
        spec.ForwardPassMode.TRAIN: contextlib.nullcontext
    }

    with contexts[mode]():
      logits_batch = model(input_batch)

    return logits_batch, None

  def output_activation_fn(self,
                           logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:

    activation_fn = {
        spec.LossType.SOFTMAX_CROSS_ENTROPY: F.softmax,
        spec.LossType.SIGMOID_CROSS_ENTROPY: F.sigmoid,
        spec.LossType.MEAN_SQUARED_ERROR: lambda z: z
    }

    return activation_fn[loss_type](logits_batch)

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(self, label_batch: spec.Tensor,
              logits_batch: spec.Tensor) -> spec.Tensor:  # differentiable

    return F.cross_entropy(logits_batch, label_batch, reduction='none')

  def _eval_metric(self, logits, labels):
    """Return the mean accuracy and loss as a dict."""
    predicted = torch.argmax(logits, 1)
    # not accuracy, but nr. of correct predictions
    accuracy = (predicted == labels).sum().item()
    loss = self.loss_fn(labels, logits).sum().item()
    n_data = len(logits)
    return {'accuracy': accuracy, 'loss': loss, 'n_data': n_data}
