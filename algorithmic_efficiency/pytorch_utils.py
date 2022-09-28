import os
from typing import Tuple

from absl import logging
import tensorflow as tf
import torch
import torch.distributed as dist

from algorithmic_efficiency.profiler import Profiler


def pytorch_setup() -> Tuple[bool, int, torch.device, int]:
  use_pytorch_ddp = 'LOCAL_RANK' in os.environ
  rank = int(os.environ['LOCAL_RANK']) if use_pytorch_ddp else 0
  device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
  n_gpus = torch.cuda.device_count()
  return use_pytorch_ddp, rank, device, n_gpus


def pytorch_init(use_pytorch_ddp: bool, rank: int, profiler: Profiler) -> None:
  # Make sure no GPU memory is preallocated to Jax.
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
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
    # only log once (for local rank == 0)
    if rank != 0:

      def logging_pass(*args):
        pass

      logging.info = logging_pass
    # initialize the process group
    dist.init_process_group('nccl')


# DO NOT SUBMIT make sure this works
def update_dropout(model, dropout_prob):
  # model.modules() returns the model itself as the first element.
  for child in list(model.modules())[1:]:
    if isinstance(child, torch.nn.Dropout):
      child.p = dropout_prob
    update_dropout(child, dropout_prob)


# DO NOT SUBMIT make sure this works
def update_attention_dropout(model, attention_dropout_prob):
  # model.modules() returns the model itself as the first element.
  for child in list(model.modules())[1:]:
    if isinstance(child, torch.nn.TransformerDecoderLayer):
      child.self_attn.dropout = attention_dropout_prob
      child.multihead_attn.dropout = attention_dropout_prob
    elif isinstance(child, torch.nn.TransformerEncoderLayer):
      child.self_attn.dropout = attention_dropout_prob
    update_dropout(child, attention_dropout_prob)
