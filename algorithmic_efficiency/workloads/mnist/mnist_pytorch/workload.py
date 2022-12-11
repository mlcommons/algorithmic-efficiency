"""MNIST workload implemented in PyTorch."""

from collections import OrderedDict
import contextlib
import itertools
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from algorithmic_efficiency import init_utils
from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
from algorithmic_efficiency.workloads.mnist.workload import BaseMnistWorkload

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


class _Model(nn.Module):

  def __init__(self) -> None:
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


def mnist_to_torch(batch: Dict[str, spec.Tensor]) -> Dict[str, spec.Tensor]:
  # Slice off the part of the batch for this device and then transpose from
  # [N, H, W, C] to [N, C, H, W]. Only transfer the inputs to GPU.
  new_batch = {}
  for k, v in batch.items():
    if USE_PYTORCH_DDP:
      new_v = v[RANK]
    else:
      new_v = v.reshape(-1, *v.shape[2:])
    if k == 'inputs':
      new_v = np.transpose(new_v, (0, 3, 1, 2))
    dtype = torch.long if k == 'targets' else torch.float
    new_batch[k] = torch.as_tensor(new_v, dtype=dtype, device=DEVICE)
  return new_batch


class MnistWorkload(BaseMnistWorkload):

  def _build_input_queue(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      global_batch_size: int,
      cache: Optional[bool] = None,
      repeat_final_dataset: Optional[bool] = None,
      num_batches: Optional[int] = None) -> Iterator[Dict[str, spec.Tensor]]:

    np_iter = super()._build_input_queue(data_rng,
                                         split,
                                         data_dir,
                                         global_batch_size,
                                         num_batches,
                                         repeat_final_dataset)
    return map(mnist_to_torch, np_iter)

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """Dropout is unused."""
    del dropout_rate
    del aux_dropout_rate
    torch.random.manual_seed(rng[0])
    model = _Model()
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
    return param_key in ['net.layer2.weight', 'net_layer2.bias']

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

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(self,
              label_batch: spec.Tensor,
              logits_batch: spec.Tensor,
              mask_batch: Optional[spec.Tensor] = None,
              label_smoothing: float = 0.0) -> Tuple[spec.Tensor, spec.Tensor]:
    """Return (correct scalar average loss, 1-d array of per-example losses)."""
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
    return summed_loss / n_valid_examples, per_example_losses

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
    weights = batch.get('weights')
    if weights is None:
      weights = torch.ones(len(logits)).to(DEVICE)
    _, predicted = torch.max(logits.data, 1)
    # Number of correct predictions.
    accuracy = ((predicted == batch['targets']) * weights).sum()
    _, per_example_losses = self.loss_fn(batch['targets'], logits, weights)
    loss = per_example_losses.sum()
    return {'accuracy': accuracy, 'loss': loss}
