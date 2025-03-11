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

  def _build_input_queue():
    pass
  
  def eval_step():
    pass
