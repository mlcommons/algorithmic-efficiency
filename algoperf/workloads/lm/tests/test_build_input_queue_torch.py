import jax
import torch

from algoperf.profiler import PassThroughProfiler
from algoperf.pytorch_utils import pytorch_init
from algoperf.pytorch_utils import pytorch_setup
from algoperf.workloads.lm.lm_pytorch.workload import LmWorkload

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


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

  print(f"RANK {RANK} of {N_GPUS}")
  sync_ddp()

  # batch = next(input_queue)
  # inputs, targets = batch['inputs'], batch['targets']
  # print(f"inputs.shape: {inputs.shape}")
  # print(f"inputs: {inputs}")

  # Start test.
  for _ in range(100):

    batch = next(input_queue)

    assert type(batch) == dict
    assert 'inputs' in batch
    assert 'targets' in batch

    inputs, targets = batch['inputs'], batch['targets']

    assert type(inputs) == torch.Tensor
    assert type(targets) == torch.Tensor

    assert inputs.device == DEVICE
    assert targets.device == DEVICE

    assert inputs.dtype == dtype
    assert targets.dtype == dtype

    assert inputs.shape == (local_batch_size, seq_len)
    assert targets.shape == (local_batch_size, seq_len)

    assert torch.equal(inputs[:, 1:], targets[:, :-1])

  print(f"=== ALL TEST PASSED ===")


def main():
  profiler = PassThroughProfiler()
  pytorch_init(USE_PYTORCH_DDP, RANK, profiler)
  test_dataloader_torch()


if __name__ == '__main__':
  main()
