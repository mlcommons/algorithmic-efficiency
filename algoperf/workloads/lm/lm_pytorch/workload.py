"""LM workload implemented in PyTorch."""

import contextlib
from typing import Dict, Iterator, Optional, Tuple

from absl import logging
import jax
import tensorflow as tf
import torch
import torch.distributed as dist
from torch.nn import DataParallel as DP
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from algoperf import param_utils
from algoperf import pytorch_utils
from algoperf import spec
from algoperf.workloads.lm.workload import BaseLmWorkload

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()


class LmWorkload(BaseLmWorkload):
  """LM PyTorch workload."""

  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         num_batches: Optional[int] = None,
                         repeat_final_dataset: bool = False) -> Iterator[Dict[str, spec.Tensor]]:
    not_train = split != 'train'
    per_device_batch_size = int(global_batch_size / N_GPUS)
  
    seq_len = 2048  # TODO: define it somewehere else
    DTYPE = torch.int32  # TODO: decide between int32 and int64.

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
        inputs = torch.as_tensor(batch['inputs'], dtype=DTYPE, device=DEVICE)  # (N_GPUS, global_batch_size, seq_len)
        targets = torch.as_tensor(batch['targets'], dtype=DTYPE, device=DEVICE)  # (N_GPUS, global_batch_size, seq_len)

        # Send batch to other devices when using DDP.
        if USE_PYTORCH_DDP:
          if not_train:
            # During eval, the batch size of the remainder might be different.
            per_device_batch_size = torch.tensor(len(targets[0]), dtype=DTYPE, device=DEVICE)
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
          per_device_batch_size = torch.empty((1,), dtype=DTYPE, device=DEVICE)
          dist.broadcast(per_device_batch_size, src=0)

        # N_GPUS - 1 since we don't broadcast the shard for RANK 0.
        inputs = torch.empty((N_GPUS-1, per_device_batch_size, seq_len), dtype=DTYPE, device=DEVICE)
        targets = torch.empty((N_GPUS-1, per_device_batch_size, seq_len), dtype=DTYPE, device=DEVICE)
        dist.broadcast(inputs, src=0)
        dist.broadcast(targets, src=0)
        # RANK - 1 since we don't broadcast the shard for RANK 0.
        inputs, targets = inputs[RANK-1], targets[RANK-1]

      if weights is None:
        weights = torch.ones(per_device_batch_size, device=DEVICE)
      batch = {
          'inputs': inputs,
          'targets': targets,
          'weights': weights,
      }
      yield batch

      
  def eval_step():
    pass
