"""Criteo1TB workload implemented in PyTorch."""
import contextlib
from typing import Dict, Optional, Tuple

import jax
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_pytorch import \
    metrics
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_pytorch.models import \
    DlrmSmall
from algorithmic_efficiency.workloads.criteo1tb.workload import \
    BaseCriteo1TbDlrmSmallWorkload

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


class Criteo1TbDlrmSmallWorkload(BaseCriteo1TbDlrmSmallWorkload):

  @property
  def model_params_types(self):
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    if self._param_types is None:
      self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    return self._param_types

  def loss_fn(self,
              label_batch: spec.Tensor,
              logits_batch: spec.Tensor,
              mask_batch: Optional[spec.Tensor] = None,
              label_smoothing: float = 0.0) -> spec.Tensor:
    del label_smoothing
    per_example_losses = metrics.per_example_sigmoid_binary_cross_entropy(
        logits=logits_batch, targets=label_batch)
    if mask_batch is not None:
      weighted_losses = per_example_losses * mask_batch
      normalization = mask_batch.sum()
    else:
      weighted_losses = per_example_losses
      normalization = label_batch.shape[0]
    return torch.sum(weighted_losses, dim=-1) / normalization

  def _eval_metric(self, logits: spec.Tensor,
                   targets: spec.Tensor) -> Dict[str, int]:
    loss = self.loss_fn(logits, targets).sum()
    return {'loss': loss}

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model = DlrmSmall(
        vocab_sizes=self.vocab_sizes,
        total_vocab_sizes=sum(self.vocab_sizes),
        num_dense_features=self.num_dense_features,
        mlp_bottom_dims=(128, 128),
        mlp_top_dims=(256, 128, 1),
        embed_dim=64)

    self._param_shapes = {
        k: spec.ShapeTuple(v.shape) for k, v in model.named_parameters()
    }
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = DDP(model, device_ids=[RANK], output_device=RANK)
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
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del rng
    del update_batch_norm

    model = params
    inputs = augmented_and_preprocessed_input_batch['inputs']

    if mode == spec.ForwardPassMode.EVAL:
      model.eval()

    if mode == spec.ForwardPassMode.TRAIN:
      model.train()

    contexts = {
        spec.ForwardPassMode.EVAL: torch.no_grad,
        spec.ForwardPassMode.TRAIN: contextlib.nullcontext
    }

    with contexts[mode]():
      logits_batch = model(inputs)

    return logits_batch, None

  def build_input_queue(self,
                        data_rng: jax.random.PRNGKey,
                        split: str,
                        data_dir: str,
                        global_batch_size: int,
                        num_batches: Optional[int] = None,
                        repeat_final_dataset: bool = False):
    per_device_batch_size = int(global_batch_size / N_GPUS)

    # Only create and iterate over tf input pipeline in one Python process to
    # avoid creating too many threads.
    # if RANK == 0:
    #   dataset_iter = super().build_input_queue(data_rng,
    #                                            split,
    #                                            data_dir,
    #                                            global_batch_size,
    #                                            num_batches,
    #                                            repeat_final_dataset)
    np_iter = super().build_input_queue(data_rng,
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
            batch['targets'], dtype=torch.float32, device=DEVICE)
        weights = torch.as_tensor(
            batch['weights'], dtype=torch.bool, device=DEVICE)

        # Send batch to other devices when using DDP.
        if USE_PYTORCH_DDP:
          # During eval, the batch size of the remainder might be different.
          if split != 'train':
            per_device_batch_size = torch.tensor(
                len(targets[0]), dtype=torch.int32, device=DEVICE)
            dist.broadcast(per_device_batch_size, src=0)
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
        # During eval, the batch size of the remainder might be different.
        if split != 'train':
          per_device_batch_size = torch.empty((1,),
                                              dtype=torch.int32,
                                              device=DEVICE)
          dist.broadcast(per_device_batch_size, src=0)

        inputs = torch.empty((N_GPUS, per_device_batch_size, 39),
                             dtype=torch.float32,
                             device=DEVICE)
        dist.broadcast(inputs, src=0)
        inputs = inputs[RANK]
        targets = torch.empty((N_GPUS, per_device_batch_size, 1),
                              dtype=torch.float32,
                              device=DEVICE)
        dist.broadcast(targets, src=0)
        targets = targets[RANK]
        weights = torch.empty((N_GPUS, per_device_batch_size, 39),
                              dtype=torch.bool,
                              device=DEVICE)
        dist.broadcast(weights, src=0)
        weights = weights[RANK]

      batch = {
          'inputs': inputs,
          'targets': targets,
          'weights': weights,
      }
      yield batch

  def _eval_batch(self, params, batch):
    logits, _ = self.model_fn(
        params,
        batch,
        model_state=None,
        mode=spec.ForwardPassMode.EVAL,
        rng=None,
        update_batch_norm=False)
    per_example_losses = metrics.per_example_sigmoid_binary_cross_entropy(
        logits, batch['targets'])
    batch_loss_numerator = torch.sum(per_example_losses)
    batch_loss_denominator = torch.sum(batch['weights'])
    return batch_loss_numerator, batch_loss_denominator
