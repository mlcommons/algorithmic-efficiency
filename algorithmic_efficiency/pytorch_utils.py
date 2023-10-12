import os
from typing import Tuple

from absl import logging
import jax
import tensorflow as tf
import torch
import torch.distributed as dist

from algorithmic_efficiency import spec
from algorithmic_efficiency.profiler import Profiler
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.models import \
    BatchNorm as ConformerBatchNorm
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_pytorch.models import \
    BatchNorm as DeepspeechBatchNorm


def pytorch_setup() -> Tuple[bool, int, torch.device, int]:
  use_pytorch_ddp = 'LOCAL_RANK' in os.environ
  rank = int(os.environ['LOCAL_RANK']) if use_pytorch_ddp else 0
  device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
  n_gpus = torch.cuda.device_count()
  return use_pytorch_ddp, rank, device, n_gpus


def pytorch_init(use_pytorch_ddp: bool, rank: int, profiler: Profiler) -> None:
  # Make sure no GPU memory is preallocated to Jax.
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  # Only use CPU for Jax to avoid memory issues.
  jax.config.update('jax_platforms', 'cpu')
  # From the docs: "(...) causes cuDNN to benchmark multiple convolution
  # algorithms and select the fastest."
  torch.backends.cudnn.benchmark = True

  if use_pytorch_ddp:
    # Avoid tf input pipeline creating too many threads.
    if rank != 0:
      tf.config.threading.set_intra_op_parallelism_threads(1)
      tf.config.threading.set_inter_op_parallelism_threads(1)

    torch.cuda.set_device(rank)
    profiler.set_local_rank(rank)
    # Only log once (for local rank == 0).
    if rank != 0:

      def logging_pass(*args):
        pass

      logging.info = logging_pass
    # Initialize the process group.
    dist.init_process_group('nccl')


def sync_ddp_time(time: float, device: torch.device) -> float:
  time_tensor = torch.tensor(time, dtype=torch.float64, device=device)
  dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
  return time_tensor.item()


def update_batch_norm_fn(module: spec.ParameterContainer,
                         update_batch_norm: bool) -> None:
  bn_layers = (
      torch.nn.modules.batchnorm._BatchNorm,  # PyTorch BN base class.
      ConformerBatchNorm,  # Custom BN class for conformer model.
      DeepspeechBatchNorm,  # Custom BN class for deepspeech model.
  )
  if isinstance(module, bn_layers):
    if not update_batch_norm:
      module.eval()
      module.momentum_backup = module.momentum
      # module.momentum can be float or torch.Tensor.
      module.momentum = 0. * module.momentum_backup
    elif hasattr(module, 'momentum_backup'):
      module.momentum = module.momentum_backup
    module.track_running_stats = update_batch_norm
