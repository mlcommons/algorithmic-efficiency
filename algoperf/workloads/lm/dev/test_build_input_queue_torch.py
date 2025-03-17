
import jax
import torch
import pdb
import numpy as np
  
from algoperf import random_utils as prng
from algoperf import spec
from algoperf.profiler import PassThroughProfiler
from algoperf.pytorch_utils import pytorch_init
from algoperf.pytorch_utils import pytorch_setup
from algoperf.workloads.lm.lm_pytorch.workload import LmWorkload

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()

n_gpus = max(N_GPUS, jax.local_device_count())

def sync_ddp():
  if torch.cuda.is_available():
    torch.cuda.synchronize()


def test_dataloader_torch():
  # Test config.
  rng_seed = 1996
  data_dir = '/fast/najroldi/data/finewebedu'
  split = 'train'
  global_batch_size = 8
  dtype = torch.int32
  seq_len = 2048

  local_batch_size = global_batch_size // N_GPUS
  
  workload = LmWorkload()

  data_rng = jax.random.PRNGKey(rng_seed)
  
  input_queue = workload._build_input_queue(
      data_rng=data_rng,
      split=split,
      data_dir=data_dir,
      global_batch_size=global_batch_size)
  
  # batch = next(input_queue)
  
  print(f"RANK {RANK} of {N_GPUS}")
  sync_ddp()

  # Start test.
  for _ in range(100):
    
    batch = next(input_queue)
    assert type(batch) == dict

    assert 'inputs' in batch
    assert 'targets' in batch

    assert type(batch['inputs']) == torch.Tensor
    assert type(batch['targets']) == torch.Tensor

    assert batch['inputs'].dtype == dtype
    assert batch['targets'].dtype == dtype

    assert batch['inputs'].shape == (local_batch_size, seq_len)
    assert batch['targets'].shape == (local_batch_size, seq_len)
    
    sync_ddp()

  print(f"=== ALL TEST PASSED ===")


def main():
  profiler = PassThroughProfiler()
  pytorch_init(USE_PYTORCH_DDP, RANK, profiler)
  test_dataloader_torch()


if __name__ == '__main__':
  main()

