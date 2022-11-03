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
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_pytorch.models import \
    DlrmSmall
from algorithmic_efficiency.workloads.criteo1tb.workload import \
    BaseCriteo1TbDlrmSmallWorkload

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


class Criteo1TbDlrmSmallWorkload(BaseCriteo1TbDlrmSmallWorkload):

  def _per_example_sigmoid_binary_cross_entropy(
      self, logits: spec.Tensor, targets: spec.Tensor) -> spec.Tensor:
    ls = torch.nn.LogSigmoid()
    log_p = ls(logits)
    log_not_p = ls(-logits)
    per_example_losses = -1.0 * (targets * log_p + (1 - targets) * log_not_p)
    per_example_losses = per_example_losses.reshape(len(per_example_losses), -1)
    return per_example_losses.sum(1)

  def loss_fn(self,
              label_batch: spec.Tensor,
              logits_batch: spec.Tensor,
              mask_batch: Optional[spec.Tensor] = None,
              label_smoothing: float = 0.0) -> spec.Tensor:
    del label_smoothing
    per_example_losses = self._per_example_sigmoid_binary_cross_entropy(
        logits=logits_batch, targets=label_batch)
    if mask_batch is not None:
      per_example_losses *= mask_batch
    return per_example_losses

  def _eval_metric(self, logits: spec.Tensor,
                   targets: spec.Tensor) -> Dict[str, int]:
    loss = self.loss_fn(logits, targets).sum()
    return {'loss': loss}

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """Dropout is unused."""
    del dropout_rate
    del aux_dropout_rate
    torch.random.manual_seed(rng[0])
    model = DlrmSmall(
        vocab_sizes=self.vocab_sizes,
        total_vocab_sizes=sum(self.vocab_sizes),
        num_dense_features=self.num_dense_features,
        mlp_bottom_dims=self.mlp_bottom_dims,
        mlp_top_dims=self.mlp_top_dims,
        embed_dim=self.embed_dim)
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
    return param_key in ['top_mlp.4.weight', 'top_mlp.4.bias']

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

  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         num_batches: Optional[int] = None,
                         repeat_final_dataset: bool = False):
    not_train = split != 'train'
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
            batch['targets'], dtype=torch.float32, device=DEVICE)
        if not_train:
          weights = torch.as_tensor(
              batch['weights'], dtype=torch.float32, device=DEVICE)

        # Send batch to other devices when using DDP.
        if USE_PYTORCH_DDP:
          # During eval, the batch size of the remainder might be different.
          if not_train:
            per_device_batch_size = torch.tensor(
                len(targets[0]), dtype=torch.int32, device=DEVICE)
            dist.broadcast(per_device_batch_size, src=0)
            dist.broadcast(weights, src=0)
            weights = weights[0]
          dist.broadcast(inputs, src=0)
          inputs = inputs[0]
          dist.broadcast(targets, src=0)
          targets = targets[0]
        else:
          inputs = inputs.view(-1, *inputs.shape[2:])
          targets = targets.view(-1, *targets.shape[2:])
          if not_train:
            weights = weights.view(-1, *weights.shape[2:])
      else:
        # During eval, the batch size of the remainder might be different.
        if not_train:
          per_device_batch_size = torch.empty((1,),
                                              dtype=torch.int32,
                                              device=DEVICE)
          dist.broadcast(per_device_batch_size, src=0)
          weights = torch.empty((N_GPUS, per_device_batch_size, 1),
                                dtype=torch.float32,
                                device=DEVICE)
          dist.broadcast(weights, src=0)
          weights = weights[RANK]

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

      batch = {
          'inputs': inputs,
          'targets': targets,
          'weights': weights,
      }
      yield batch

  def _eval_batch(self,
                  params: spec.ParameterContainer,
                  batch: Dict[str, spec.Tensor]) -> spec.Tensor:
    logits, _ = self.model_fn(
        params,
        batch,
        model_state=None,
        mode=spec.ForwardPassMode.EVAL,
        rng=None,
        update_batch_norm=False)
    weights = batch.get('weights')
    if weights is None:
      weights = torch.ones(len(logits), device=DEVICE)
    per_example_losses = self.loss_fn(logits, batch['targets'], weights)
    loss = per_example_losses.sum()
    return loss
