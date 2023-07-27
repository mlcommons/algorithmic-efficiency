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

  @property
  def eval_batch_size(self) -> int:
    return 262_144

  def _per_example_sigmoid_binary_cross_entropy(
      self, logits: spec.Tensor, targets: spec.Tensor) -> spec.Tensor:
    ls = torch.nn.LogSigmoid()
    log_p = ls(logits)
    log_not_p = ls(-logits)
    per_example_losses = -1.0 * (targets * log_p + (1 - targets) * log_not_p)
    per_example_losses = per_example_losses.reshape(len(per_example_losses), -1)
    return per_example_losses.sum(1)

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,  # Dense (not one-hot) labels.
      logits_batch: spec.Tensor,
      mask_batch: Optional[spec.Tensor] = None,
      label_smoothing: float = 0.0) -> Dict[str, spec.Tensor]:  # differentiable
    """Evaluate the (masked) loss function at (label_batch, logits_batch).

    Return {'summed': scalar summed loss, 'n_valid_examples': scalar number of
    valid examples in batch, 'per_example': 1-d array of per-example losses}
    (not synced across devices).
    """
    del label_smoothing
    batch_size = label_batch.shape[0]
    label_batch = torch.reshape(label_batch, (batch_size,))
    logits_batch = torch.reshape(logits_batch, (batch_size,))
    per_example_losses = self._per_example_sigmoid_binary_cross_entropy(
        logits=logits_batch, targets=label_batch)
    if mask_batch is not None:
      mask_batch = torch.reshape(mask_batch, (batch_size,))
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

  def _eval_metric(self, logits: spec.Tensor,
                   targets: spec.Tensor) -> Dict[str, int]:
    summed_loss = self.loss_fn(logits, targets)['summed']
    return {'loss': summed_loss}

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """Only dropout is used."""
    del aux_dropout_rate
    torch.random.manual_seed(rng[0])
    model = DlrmSmall(
        vocab_size=self.vocab_size,
        num_dense_features=self.num_dense_features,
        mlp_bottom_dims=self.mlp_bottom_dims,
        mlp_top_dims=self.mlp_top_dims,
        embed_dim=self.embed_dim,
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
        spec.ForwardPassMode.TRAIN: contextlib.nullcontext,
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
    weights = None
    while True:
      if RANK == 0:
        batch = next(np_iter)  # pylint: disable=stop-iteration-return
        inputs = torch.as_tensor(
            batch['inputs'], dtype=torch.float32, device=DEVICE)
        targets = torch.as_tensor(
            batch['targets'], dtype=torch.float32, device=DEVICE)
        if not_train:
          weights = batch.get('weights')
          if weights is None:
            weights = torch.ones((N_GPUS, per_device_batch_size, 1),
                                 dtype=torch.float32,
                                 device=DEVICE)
          else:
            weights = torch.as_tensor(
                weights, dtype=torch.float32, device=DEVICE)
        # Send batch to other devices when using DDP.
        if USE_PYTORCH_DDP:
          if not_train:
            # During eval, the batch size of the remainder might be different.
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
        if not_train:
          # During eval, the batch size of the remainder might be different.
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

      if weights is None:
        weights = torch.ones(per_device_batch_size, device=DEVICE)
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
    summed_loss = self.loss_fn(
        label_batch=batch['targets'], logits_batch=logits,
        mask_batch=weights)['summed']
    return summed_loss


class Criteo1TbDlrmSmallTestWorkload(Criteo1TbDlrmSmallWorkload):
  vocab_size: int = 32 * 128 * 16
