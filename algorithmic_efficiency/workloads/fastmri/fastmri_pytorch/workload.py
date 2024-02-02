"""FastMRI workload implemented in PyTorch."""

import contextlib
import math
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import pytorch_utils
from algorithmic_efficiency import spec
import algorithmic_efficiency.random_utils as prng
from algorithmic_efficiency.workloads.fastmri.fastmri_pytorch.models import \
    UNet
from algorithmic_efficiency.workloads.fastmri.fastmri_pytorch.ssim import ssim
from algorithmic_efficiency.workloads.fastmri.workload import \
    BaseFastMRIWorkload

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()


class FastMRIWorkload(BaseFastMRIWorkload):

  def _build_input_queue(self,
                         data_rng: spec.RandomState,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         cache: Optional[bool] = None,
                         repeat_final_dataset: Optional[bool] = None,
                         num_batches: Optional[int] = None):
    per_device_batch_size = int(global_batch_size / N_GPUS)

    # Only create and iterate over tf input pipeline in one Python process to
    # avoid creating too many threads.
    if RANK == 0:
      data_rng = data_rng.astype('uint32')
      np_iter = super()._build_input_queue(data_rng,
                                           split,
                                           data_dir,
                                           global_batch_size,
                                           cache,
                                           repeat_final_dataset,
                                           num_batches)

    while True:
      if RANK == 0:
        batch = next(np_iter)  # pylint: disable=stop-iteration-return
        tensor_list, aux_tensor_list = [], []
        for key, value in batch.items():
          tensor = torch.as_tensor(value, device=DEVICE)
          if key == 'weights':
            weights = tensor.clone()
          else:
            if tensor.dim() == 4:
              tensor_list.append(tensor)
            else:
              aux_tensor_list.append(tensor)
          batch[key] = (
              tensor[0] if USE_PYTORCH_DDP else tensor.view(
                  -1, *value.shape[2:]))
        # Send batch to other devices when using DDP.
        if USE_PYTORCH_DDP:
          if split != 'train':
            # During eval, the batch size of the remainder might be different.
            per_device_batch_size = torch.tensor(
                len(batch['inputs']), dtype=torch.int32, device=DEVICE)
            dist.broadcast(per_device_batch_size, src=0)
            weights = weights if 'weights' in batch else None
            if weights is None:
              weights = torch.ones((N_GPUS, per_device_batch_size),
                                   dtype=torch.float64,
                                   device=DEVICE)
              # Has no effect, but without it `batch` has no `weights` key
              # for RANK == 0, but has one for all others.
              batch['weights'] = weights[0]
            dist.broadcast(weights, src=0)
          dist.broadcast(torch.stack(tensor_list), src=0)
          dist.broadcast(torch.stack(aux_tensor_list), src=0)
      else:
        batch = {}
        if split != 'train':
          # During eval, the batch size of the remainder might be different.
          per_device_batch_size = torch.empty((),
                                              dtype=torch.int32,
                                              device=DEVICE)
          dist.broadcast(per_device_batch_size, src=0)
          weights = torch.empty((N_GPUS, per_device_batch_size),
                                dtype=torch.float64,
                                device=DEVICE)
          dist.broadcast(weights, src=0)
          batch['weights'] = weights[RANK]
        tensors = torch.empty((2, N_GPUS, per_device_batch_size, 320, 320),
                              device=DEVICE)
        dist.broadcast(tensors, src=0)
        aux_tensors = torch.empty((3, N_GPUS, per_device_batch_size),
                                  device=DEVICE)
        dist.broadcast(aux_tensors, src=0)
        # Note that the batch dict in the RANK == 0 process is ordered.
        batch['inputs'] = tensors[0][RANK]
        batch['targets'] = tensors[1][RANK]
        batch['mean'] = aux_tensors[0][RANK]
        batch['std'] = aux_tensors[1][RANK]
        batch['volume_max'] = aux_tensors[2][RANK]
      yield batch

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    del aux_dropout_rate
    torch.random.manual_seed(rng)
    model = UNet(
        num_pool_layers=self.num_pool_layers,
        num_channels=self.num_channels,
        use_tanh=self.use_tanh,
        use_layer_norm=self.use_layer_norm,
        dropout_rate=dropout_rate)
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
    return param_key in ['up_conv.3.1.weight', 'up_conv.3.1.bias']

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

    if mode == spec.ForwardPassMode.TRAIN:
      model.train()

    contexts = {
        spec.ForwardPassMode.EVAL: torch.no_grad,
        spec.ForwardPassMode.TRAIN: contextlib.nullcontext,
    }

    with contexts[mode]():
      logit_batch = model(
          augmented_and_preprocessed_input_batch['inputs'].unsqueeze(
              1)).squeeze(1)

    return logit_batch, None

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
    del label_smoothing
    per_example_losses = F.l1_loss(
        logits_batch, label_batch, reduction='none').mean(dim=(1, 2))
    # mask_batch is assumed to be shape [batch].
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

  def _eval_model(self,
                  params: spec.ParameterContainer,
                  batch: Dict[str, spec.Tensor],
                  rng: spec.RandomState) -> Dict[str, spec.Tensor]:
    """Return the SSIM and loss as a dict."""
    outputs, _ = self.model_fn(
        params,
        batch,
        None,
        spec.ForwardPassMode.EVAL,
        rng,
        update_batch_norm=False)
    targets = batch['targets']
    weights = batch.get('weights')
    if weights is None:
      weights = torch.ones(len(outputs), device=DEVICE)
    weights_sum = weights.sum().to(torch.int)
    ssim_sum = ssim(
        outputs[:weights_sum],
        targets[:weights_sum],
        mean=batch['mean'][:weights_sum],
        std=batch['std'][:weights_sum],
        volume_max=batch['volume_max'][:weights_sum]).sum()
    summed_loss = self.loss_fn(targets, outputs, weights)['summed']
    return {'ssim': ssim_sum, 'loss': summed_loss}

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str,
                           global_step: int = 0) -> Dict[str, float]:
    """Run a full evaluation of the model."""
    del model_state
    del global_step
    data_rng, model_rng = prng.split(rng, 2)
    num_batches = int(math.ceil(num_examples / global_batch_size))
    if split not in self._eval_iters:
      # These iterators repeat indefinitely.
      self._eval_iters[split] = self._build_input_queue(
          data_rng,
          split,
          data_dir,
          global_batch_size=global_batch_size,
          repeat_final_dataset=True,
          num_batches=num_batches)

    total_metrics = {
        'ssim': torch.tensor(0., device=DEVICE),
        'loss': torch.tensor(0., device=DEVICE),
    }
    for _ in range(num_batches):
      batch = next(self._eval_iters[split])
      batch_metrics = self._eval_model(params, batch, model_rng)
      total_metrics = {
          k: v + batch_metrics[k] for k, v in total_metrics.items()
      }
    if USE_PYTORCH_DDP:
      for metric in total_metrics.values():
        dist.all_reduce(metric)
    return {k: float(v.item() / num_examples) for k, v in total_metrics.items()}


class FastMRIModelSizeWorkload(FastMRIWorkload):

  @property
  def num_pool_layers(self) -> bool:
    """Whether or not to use tanh activations in the model."""
    return 3

  @property
  def num_channels(self) -> bool:
    """Whether or not to use tanh activations in the model."""
    return 64

  @property
  def validation_target_value(self) -> float:
    return 0.723559

  @property
  def test_target_value(self) -> float:
    return 0.740726


class FastMRITanhWorkload(FastMRIWorkload):

  @property
  def use_tanh(self) -> bool:
    """Whether or not to use tanh activations in the model."""
    return True

  @property
  def validation_target_value(self) -> float:
    return 0.717840

  @property
  def test_target_value(self) -> float:
    return 0.734505


class FastMRILayerNormWorkload(FastMRIWorkload):

  @property
  def use_layer_norm(self) -> bool:
    """Whether or not to use tanh activations in the model."""
    return True

  @property
  def validation_target_value(self) -> float:
    return 0.723284

  @property
  def test_target_value(self) -> float:
    return 0.739996
