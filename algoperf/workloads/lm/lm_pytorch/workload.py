"""LM workload implemented in PyTorch."""

import contextlib
from typing import Any, Dict, Optional, Tuple

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
  
  def init_model_fn():
    pass

  def model_fn():
    pass

  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         num_batches: Optional[int] = None,
                         repeat_final_dataset: bool = False):
    per_device_batch_size = int(global_batch_size / N_GPUS)

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
    while True:
      if RANK == 0:
        batch = next(np_iter)
        inputs = torch.as_tensor(
            batch['inputs'], dtype=torch.float32, device=DEVICE)
        targets = torch.as_tensor(
            batch['targets'], dtype=torch.float32, device=DEVICE)
        # Send batch to other devices when using DDP.
        if USE_PYTORCH_DDP:
          dist.broadcast(inputs, src=0)
          inputs = inputs[0]  # TODO: check 
          dist.broadcast(targets, src=0)
          targets = targets[0]  # TODO: check
      else:
        batch = {}
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
          # 'weights': weights,
      }
      yield batch

      
  def eval_step():
    pass
