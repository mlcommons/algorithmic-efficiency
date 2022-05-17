"""MNIST workload implemented in PyTorch."""
from collections import OrderedDict
import contextlib
import itertools
from typing import Any, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torchvision import transforms
from torchvision.datasets import MNIST

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.mnist.workload import BaseMnistWorkload

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class _Model(nn.Module):

  def __init__(self):
    super().__init__()
    input_size = 28 * 28
    num_hidden = 128
    num_classes = 10
    self.net = nn.Sequential(
        OrderedDict([('layer1',
                      torch.nn.Linear(input_size, num_hidden, bias=True)),
                     ('layer1_sig', torch.nn.Sigmoid()),
                     ('layer2',
                      torch.nn.Linear(num_hidden, num_classes, bias=True)),
                     ('output', torch.nn.LogSoftmax(dim=1))]))

  def forward(self, x: spec.Tensor):
    x = x.view(x.size()[0], -1)
    return self.net(x)


class DictMNIST(MNIST):

  def __getitem__(self, index: int) -> Dict[str, Any]:
    image, label = super().__getitem__(index)
    return {'inputs': image, 'targets': label}


class MnistWorkload(BaseMnistWorkload):

  def _build_dataset(self,
                     data_rng: spec.RandomState,
                     split: str,
                     data_dir: str,
                     batch_size: int):

    dataloader_split = 'train' if split == 'eval_train' else split
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((self.train_mean,), (self.train_stddev,))
    ])
    dataset = DictMNIST(
        data_dir, train=dataloader_split, download=True, transform=transform)
    if split != 'test':
      if split in ['train', 'validation']:
        train_dataset, validation_dataset = data_utils.random_split(
            dataset,
            [self.num_train_examples, self.num_validation_examples],
            generator=torch.Generator().manual_seed(int(data_rng[0])))
        if split == 'train':
          dataset = train_dataset
        elif split == 'validation':
          dataset = validation_dataset
      if split == 'eval_train':
        dataset, _ = data_utils.random_split(
            dataset,
            [self.num_eval_train_examples,
             60000 - self.num_eval_train_examples],
            generator=torch.Generator().manual_seed(int(data_rng[0])))
    # TODO: set seeds properly
    is_train = split == 'train'
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train, pin_memory=True)
    if is_train:
      dataloader = itertools.cycle(dataloader)

    return dataloader

  @property
  def model_params_types(self):
    """The shapes of the parameters in the workload model."""
    if self._param_types is None:
      self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    return self._param_types

  # Return whether or not a key in spec.ParameterContainer is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def build_input_queue(self,
                        data_rng,
                        split: str,
                        data_dir: str,
                        global_batch_size: int):
    return self._build_dataset(data_rng, split, data_dir, global_batch_size)

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model = _Model()
    self._param_shapes = {
        k: spec.ShapeTuple(v.shape) for k, v in model.named_parameters()
    }
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    model.to(DEVICE)
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
    del update_batch_norm

    model = params

    if mode == spec.ForwardPassMode.EVAL:
      model.eval()

    contexts = {
        spec.ForwardPassMode.EVAL: torch.no_grad,
        spec.ForwardPassMode.TRAIN: contextlib.nullcontext
    }

    with contexts[mode]():
      logits_batch = model(augmented_and_preprocessed_input_batch['inputs'])

    return logits_batch, None

  # TODO(znado): Implement.
  # Keep this separate from the loss function in order to support optimizers
  # that use the logits.
  def output_activation_fn(self,
                           logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:
    raise NotImplementedError

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(self, label_batch: spec.Tensor,
              logits_batch: spec.Tensor) -> spec.Tensor:  # differentiable

    return F.nll_loss(logits_batch, label_batch, reduction='none')

  def _eval_model(
      self,
      params: spec.ParameterContainer,
      batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      rng: spec.RandomState) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    """Return the mean accuracy and loss as a dict."""
    logits, _ = self.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.EVAL,
        rng,
        update_batch_norm=False)
    _, predicted = torch.max(logits.data, 1)
    # Number of correct predictions.
    accuracy = (predicted == batch['targets']).sum().item()
    loss = self.loss_fn(batch['targets'], logits).sum().item()
    num_data = len(logits)
    return {'accuracy': accuracy, 'loss': loss, 'num_data': num_data}
