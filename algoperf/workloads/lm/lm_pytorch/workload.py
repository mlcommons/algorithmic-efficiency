"""LM workload implemented in PyTorch."""

from typing import Dict, Iterator, Optional, Tuple

import jax
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from algoperf import param_utils
from algoperf import pytorch_utils
from algoperf import spec
from algoperf.workloads.lm.workload import BaseLmWorkload
from algoperf.workloads.lm.lm_pytorch.models import LinearLayer

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()


class LmWorkload(BaseLmWorkload):
  """LM PyTorch workload."""

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    
    if hasattr(self, '_model'):
        self._model.reset_parameters()
        return self._model, None

    torch.manual_seed(rng[0])
    self._model = LinearLayer(vocab_size=self._vocab_size)
    self._param_shapes = param_utils.pytorch_param_shapes(self._model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    self._model.to(DEVICE)
    
    if N_GPUS > 1:
        if USE_PYTORCH_DDP:
            self._model = DDP(self._model, device_ids=[RANK], output_device=RANK)
        else:
            self._model = torch.nn.DataParallel(self._model)
            
    return self._model, None

  def model_fn(
      self,
      params: spec.ParameterContainer,
      batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    
    del model_state, rng, update_batch_norm  # Not used for linear model
    model = params
    inputs = batch['inputs'].float()  # Convert one-hot to float
    logits = model(inputs)
    return logits, None

  def _build_input_queue(
      self,
      data_rng: jax.random.PRNGKey,
      split: str,
      data_dir: str,
      global_batch_size: int,
      num_batches: Optional[int] = None,
      repeat_final_dataset: bool = False) -> Iterator[Dict[str, spec.Tensor]]:
    not_train = split != 'train'
    per_device_batch_size = int(global_batch_size / N_GPUS)

    seq_len = self._seq_len  # TODO: define it somewehere else?
    dtype = torch.int32  # TODO: decide between int32 and int64.

    # Only create and iterate over tf input pipeline in one Python process to
    # avoid creating too many threads.
    if RANK == 0:
      np_iter = super()._build_input_queue(
          data_rng=data_rng,
          split=split,
          data_dir=data_dir,
          global_batch_size=global_batch_size,
          num_batches=num_batches,
          repeat_final_dataset=repeat_final_dataset)
    weights = None

    while True:
      # Only iterate over tf input pipeline in one Python process to
      # avoid creating too many threads.
      if RANK == 0:
        batch = next(np_iter)  # pylint: disable=stop-iteration-return
        inputs = torch.as_tensor(
            batch['inputs'], dtype=dtype,
            device=DEVICE)  # (N_GPUS, global_batch_size, seq_len)
        targets = torch.as_tensor(
            batch['targets'], dtype=dtype,
            device=DEVICE)  # (N_GPUS, global_batch_size, seq_len)

        # Send batch to other devices when using DDP.
        if USE_PYTORCH_DDP:
          if not_train:
            # During eval, the batch size of the remainder might be different.
            per_device_batch_size = torch.tensor(
                len(targets[0]), dtype=dtype, device=DEVICE)
            dist.broadcast(per_device_batch_size, src=0)
          # We don't broadcast the shard for RANK 0.
          dist.broadcast(inputs[1:], src=0)
          dist.broadcast(targets[1:], src=0)

        # RANK 0 extracts his shard. If not DDP, this just flattens.
        inputs, targets = inputs[0], targets[0]

      else:
        # Receive batch from rank 0.
        if not_train:
          # During eval, the batch size of the remainder might be different.
          per_device_batch_size = torch.empty((1,), dtype=dtype, device=DEVICE)
          dist.broadcast(per_device_batch_size, src=0)

        # N_GPUS - 1 since we don't broadcast the shard for RANK 0.
        inputs = torch.empty((N_GPUS - 1, per_device_batch_size, seq_len),
                             dtype=dtype,
                             device=DEVICE)
        targets = torch.empty((N_GPUS - 1, per_device_batch_size, seq_len),
                              dtype=dtype,
                              device=DEVICE)
        dist.broadcast(inputs, src=0)
        dist.broadcast(targets, src=0)
        # RANK - 1 since we don't broadcast the shard for RANK 0.
        inputs, targets = inputs[RANK - 1], targets[RANK - 1]

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
                  batch: Dict[str, spec.Tensor],
                  model_state: spec.ModelAuxiliaryState,
                  rng: spec.RandomState) -> spec.Tensor:
    """Evaluate the model on a single batch."""
    pass
