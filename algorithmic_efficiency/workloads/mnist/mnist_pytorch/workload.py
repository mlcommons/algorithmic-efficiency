"""MNIST workload implemented in PyTorch."""

from collections import OrderedDict
import contextlib
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from torch import nn
import torch.distributed as dist
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
    del cache
    per_device_batch_size = int(global_batch_size / N_GPUS)

    # Only create and iterate over tf input pipeline in one Python process to
    # avoid creating too many threads.
    if RANK == 0:
      np_iter = super()._build_input_queue(data_rng,
                                           split,
                                           data_dir,
                                           global_batch_size,
                                           num_batches,
                                           repeat_final_dataset)
    while True:
      if RANK == 0:
        batch = next(np_iter)  # pylint: disable=stop-iteration-return
        inputs = torch.as_tensor(
            batch['inputs'], dtype=torch.float32, device=DEVICE)
        targets = torch.as_tensor(
            batch['targets'], dtype=torch.long, device=DEVICE)
        if 'weights' in batch:
          weights = torch.as_tensor(
              batch['weights'], dtype=torch.bool, device=DEVICE)
        else:
          weights = torch.ones((batch['targets'].shape[-1],),
                               dtype=torch.bool,
                               device=DEVICE)
        # Send batch to other devices when using DDP.
        if USE_PYTORCH_DDP:
          dist.broadcast(inputs, src=0)
          inputs = inputs[0]
          dist.broadcast(targets, src=0)
          targets = targets[0]
          dist.broadcast(weights, src=0)
          weights = weights[0]
        else:
          inputs = inputs.view(-1, *inputs.shape[2:])
          targets = targets.view(-1, *targets.shape[2:])
          weights = weights.view(-1, *weights.shape[2:])
      else:
        inputs = torch.empty((N_GPUS, per_device_batch_size, 28, 28, 1),
                             dtype=torch.float32,
                             device=DEVICE)
        dist.broadcast(inputs, src=0)
        inputs = inputs[RANK]
        targets = torch.empty((N_GPUS, per_device_batch_size),
                              dtype=torch.long,
                              device=DEVICE)
        dist.broadcast(targets, src=0)
        targets = targets[RANK]
        weights = torch.empty((N_GPUS, per_device_batch_size),
                              dtype=torch.bool,
                              device=DEVICE)
        dist.broadcast(weights, src=0)
        weights = weights[RANK]

      batch = {
          'inputs': inputs.permute(0, 3, 1, 2),
          'targets': targets,
          'weights': weights,
      }
      yield batch

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """Dropout is unused."""
    del dropout_rate
    del aux_dropout_rate

    if hasattr(self, '_model'):
      if isinstance(self._model, (DDP, torch.nn.DataParallel)):
        self._model.module.reset_parameters()
      else:
        self._model.reset_parameters()
      return self._model, None

    torch.random.manual_seed(rng[0])
    self._model = _Model()
    self._param_shapes = param_utils.pytorch_param_shapes(self._model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    self._model.to(DEVICE)
    self._model = torch.compile(self._model)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        self._model = DDP(self._model, device_ids=[RANK], output_device=RANK)
      else:
        self._model = torch.nn.DataParallel(self._model)
    return self._model, None

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
        'n_valid_examples': torch.as_tensor(n_valid_examples, device=DEVICE),
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
      weights = torch.ones(len(logits), device=DEVICE)
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
