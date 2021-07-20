"""MNIST workload implemented in PyTorch."""

import contextlib
import itertools
from typing import Tuple
from collections import OrderedDict

from workloads.mnist.workload import Mnist

import spec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

class _Model(nn.Module):

    def __init__(self):
      super(_Model, self).__init__()
      input_size = 28 * 28
      num_hidden = 128
      num_classes = 10
      self.net = nn.Sequential(OrderedDict([
          ('layer1',     torch.nn.Linear(input_size, num_hidden, bias=True)),
          ('layer1_sig', torch.nn.Sigmoid()),
          ('layer2',     torch.nn.Linear(num_hidden, num_classes, bias=True)),
          ('output',     torch.nn.LogSoftmax(dim=1))
      ]))

    def forward(self, x: spec.Tensor):
      return self.net(x)


class MnistWorkload(Mnist):

  def __init__(self):
    self._eval_ds = None

  def _build_dataset(self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      batch_size: int):

    assert split in ['train', 'test']
    is_train = split == 'train'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((self.train_mean,), (self.train_stddev,))
    ])
    dataset = MNIST(data_dir, train=is_train, download=True, transform=transform)
    # TODO: set seeds properly
    dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=is_train,
      pin_memory=True)

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
    pass

  # Return whether or not a key in spec.ParameterContainer is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

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
    N = raw_input_batch.size()[0]
    raw_input_batch = raw_input_batch.view(N, -1)
    return (raw_input_batch.to(DEVICE), raw_label_batch.to(DEVICE))

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model = _Model().to(DEVICE)
    return model, None

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
    del update_batch_norm

    model = params

    if mode == spec.ForwardPassMode.EVAL:
        model.eval()

    contexts = {
      spec.ForwardPassMode.EVAL: torch.no_grad,
      spec.ForwardPassMode.TRAIN: contextlib.nullcontext
    }

    with contexts[mode]():
      logits_batch = model(input_batch)

    return logits_batch, None

  # TODO(znado): Implement.
  # Keep this separate from the loss function in order to support optimizers
  # that use the logits.
  def output_activation_fn(
      self,
      logits_batch: spec.Tensor,
      loss_type: spec.LossType) -> spec.Tensor:
    raise NotImplementedError

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
