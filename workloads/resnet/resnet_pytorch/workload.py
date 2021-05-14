"""Resnet workload implemented in PyTorch."""
import contextlib
import itertools
import os
from collections import OrderedDict
from typing import Tuple

from torchvision.datasets.folder import ImageFolder

import spec
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageNet
from workloads.resnet.resnet_pytorch.resnet import resnet50
from workloads.resnet.workload import Resnet
from .utils import fast_collate

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'


class ResnetWorkload(Resnet):

  def __init__(self):
    self._eval_ds = None

  def _build_dataset(self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      batch_size: int):

    is_train = (split == "train")

    transform_config = {
      "train": transforms.Compose([
        transforms.RandomResizedCrop(
          self.center_crop_size,
          scale=self.scale_ratio_range,
          ratio=self.aspect_ratio_range),
        transforms.RandomHorizontalFlip(),
        ]),
      "test": transforms.Compose([
        transforms.Resize(self.resize_size),
        transforms.CenterCrop(self.center_crop_size),
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
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=is_train,
        collate_fn=fast_collate
    )

    if is_train:
      dataloader = itertools.cycle(dataloader)

    return dataloader


  def build_input_queue(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      batch_size: int):
    return iter(self._build_dataset(data_rng, split, data_dir, batch_size))

  @property
  def param_shapes(self):
    """
    TODO: return shape tuples from model as a tree
    """
    raise NotImplementedError

  def model_params_types(self):
    raise NotImplementedError

  # Return whether or not a key in spec.ParameterTree is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    raise NotImplementedError

  def preprocess_for_train(
      self,
      selected_raw_input_batch: spec.Tensor,
      selected_label_batch: spec.Tensor,
      train_mean: spec.Tensor,
      train_stddev: spec.Tensor,
      rng: spec.RandomState) -> spec.Tensor:
    del rng
    return self.preprocess_for_eval(
        selected_raw_input_batch, selected_label_batch, None, None)

  def preprocess_for_eval(
      self,
      raw_input_batch: spec.Tensor,
      raw_label_batch: spec.Tensor,
      train_mean: spec.Tensor,
      train_stddev: spec.Tensor) -> spec.Tensor:
    del train_mean
    del train_stddev
    return (raw_input_batch.float().to(DEVICE), raw_label_batch.to(DEVICE))

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model = resnet50().to(DEVICE)
    return model, None

  def model_fn(
      self,
      params: spec.ParameterTree,
      augmented_and_preprocessed_input_batch: spec.Tensor,
      model_state: spec.ModelAuxillaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxillaryState]:
    del model_state
    del rng
    del update_batch_norm

    model = params

    if mode == spec.ForwardPassMode.EVAL:
        model.eval()

    contexts = {
      spec.ForwardPassMode.EVAL: torch.no_grad,
      spec.ForwardPassMode.TRAIN: contextlib.nullcontext
    }

    with contexts[mode]():
      logits_batch = model(augmented_and_preprocessed_input_batch)

    return logits_batch, None

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,
      logits_batch: spec.Tensor) -> spec.Tensor:  # differentiable

    return F.nll_loss(logits_batch, label_batch)

  def _eval_metric(self, logits, labels):
    """Return the mean accuracy and loss as a dict."""
    _, predicted = torch.max(logits.data, 1)
    accuracy = (predicted == labels).cpu().numpy().mean()
    loss = self.loss_fn(labels, logits).cpu().numpy().mean()
    return {'accuracy': accuracy, 'loss': loss}
