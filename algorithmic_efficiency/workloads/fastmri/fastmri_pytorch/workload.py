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
          if tensor.dim() == 4:
            tensor_list.append(tensor)
          else:
            aux_tensor_list.append(tensor)
          batch[key] = (
              tensor[0] if USE_PYTORCH_DDP else tensor.view(
                  -1, *value.shape[2:]))
        # Send batch to other devices when using DDP.
        if USE_PYTORCH_DDP:
          # During eval, the batch size of the remainder might be different.
          if split != 'train':
            per_device_batch_size = torch.tensor(
                len(batch['inputs']), dtype=torch.int32, device=DEVICE)
            dist.broadcast(per_device_batch_size, src=0)
            weights = aux_tensor_list.pop(-1)
            dist.broadcast(weights, src=0)
          dist.broadcast(torch.stack(tensor_list), src=0)
          dist.broadcast(torch.stack(aux_tensor_list), src=0)
      else:
        batch = {}
        # During eval, the batch size of the remainder might be different.
        if split != 'train':
          per_device_batch_size = torch.empty((1,),
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
    torch.random.manual_seed(rng[0])
    model = UNet(dropout_rate=dropout_rate)
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
        spec.ForwardPassMode.TRAIN: contextlib.nullcontext
    }

    with contexts[mode]():
      logit_batch = model(
          augmented_and_preprocessed_input_batch['inputs'].unsqueeze(
              1)).squeeze(1)

    return logit_batch, None

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(self,
              label_batch: spec.Tensor,
              logits_batch: spec.Tensor,
              mask_batch: Optional[spec.Tensor] = None,
              label_smoothing: float = 0.0) -> spec.Tensor:  # differentiable
    del label_smoothing
    losses = F.l1_loss(
        logits_batch, label_batch, reduction='none').mean(dim=(1, 2))
    # mask_batch is assumed to be shape [batch].
    if mask_batch is not None:
      losses *= mask_batch
    return losses

  def _eval_model(self, params, batch, rng):
    """Return the SSIM and loss as a dict."""
    outputs, _ = self.model_fn(
        params,
        batch,
        None,
        spec.ForwardPassMode.EVAL,
        rng,
        update_batch_norm=False)
    ssim_sum = ssim(
        outputs,
        batch['targets'],
        mean=batch['mean'],
        std=batch['std'],
        volume_max=batch['volume_max']).sum()
    loss = self.loss_fn(batch['targets'], outputs).sum()
    return {'ssim': ssim_sum, 'loss': loss, 'weight': batch['weights'].sum()}

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
    del model_state
    del global_step
    data_rng, model_rng = prng.split(rng, 2)
    if split not in self._eval_iters:
      # These iterators repeat indefinitely.
      self._eval_iters[split] = self._build_input_queue(
          data_rng,
          split,
          data_dir,
          global_batch_size=global_batch_size,
          repeat_final_dataset=True)

    total_metrics = {
        'ssim': torch.tensor(0., device=DEVICE),
        'loss': torch.tensor(0., device=DEVICE),
        'weight': torch.tensor(0., device=DEVICE),
    }
    num_batches = int(math.ceil(num_examples / global_batch_size))
    for _ in range(num_batches):
      batch = next(self._eval_iters[split])
      batch_metrics = self._eval_model(params, batch, model_rng)
      total_metrics = {
          k: v + batch_metrics[k] for k, v in total_metrics.items()
      }
    if USE_PYTORCH_DDP:
      for metric in total_metrics.values():
        dist.all_reduce(metric)
    num_examples = total_metrics.pop('weight')
    return {k: float(v.item() / num_examples) for k, v in total_metrics.items()}
