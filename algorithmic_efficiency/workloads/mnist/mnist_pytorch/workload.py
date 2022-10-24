"""MNIST workload implemented in PyTorch."""
from collections import OrderedDict
import contextlib
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data as pytorch_data_utils
from torchvision import transforms
from torchvision.datasets import MNIST

from algorithmic_efficiency import data_utils
from algorithmic_efficiency import init_utils
from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
from algorithmic_efficiency.workloads.mnist.workload import BaseMnistWorkload

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


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
                      torch.nn.Linear(num_hidden, num_classes, bias=True))]))

  def reset_parameters(self) -> None:
    for m in self.net.modules():
      if isinstance(m, nn.Linear):
        init_utils.pytorch_default_init(m)

  def forward(self, x: spec.Tensor) -> spec.Tensor:
    x = x.view(x.size()[0], -1)
    return self.net(x)


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
    dataset = MNIST(
        data_dir, train=dataloader_split, download=True, transform=transform)
    if split != 'test':
      if split in ['train', 'validation']:
        train_dataset, validation_dataset = pytorch_data_utils.random_split(
            dataset,
            [self.num_train_examples, self.num_validation_examples],
            generator=torch.Generator().manual_seed(int(data_rng[0])))
        if split == 'train':
          dataset = train_dataset
        elif split == 'validation':
          dataset = validation_dataset
      if split == 'eval_train':
        dataset, _ = pytorch_data_utils.random_split(
            dataset,
            [self.num_eval_train_examples,
             60000 - self.num_eval_train_examples],
            generator=torch.Generator().manual_seed(int(data_rng[0])))
    # TODO: set seeds properly
    is_train = split == 'train'

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
        num_workers=0,
        pin_memory=True,
        drop_last=is_train)
    dataloader = data_utils.cycle(dataloader, custom_sampler=USE_PYTORCH_DDP)

    return dataloader

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key in ['net.layer2.weight', 'net_layer2.bias']

  def _build_input_queue(self,
                         data_rng,
                         split: str,
                         data_dir: str,
                         global_batch_size: int) -> Dict[str, Any]:
    it = self._build_dataset(data_rng, split, data_dir, global_batch_size)
    for batch in it:
      yield {
          'inputs': batch['inputs'].to(DEVICE, non_blocking=True),
          'targets': batch['targets'].to(DEVICE, non_blocking=True),
      }

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model = _Model()
    self._param_shapes = param_utils.pytorch_param_shapes(model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = DDP(
            model,
            device_ids=[RANK],
            output_device=RANK,
            gradient_as_bucket_view=True)
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
      dropout_rate: Optional[float],
      aux_dropout_rate: Optional[float],
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    """Dropout is unused."""
    del model_state
    del rng
    del dropout_rate
    del aux_dropout_rate
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
        dropout_rate=None,
        aux_dropout_rate=None,
        update_batch_norm=False)
    _, predicted = torch.max(logits.data, 1)
    # Number of correct predictions.
    accuracy = (predicted == batch['targets']).sum()
    loss = self.loss_fn(batch['targets'], logits).sum()
    return {'accuracy': accuracy, 'loss': loss}
