
# TODO: redo with pmap!!

import os
import jax
import tensorflow as tf
import torch
import pdb
import numpy as np
  
from algoperf import random_utils as prng
from algoperf import spec
from algoperf.profiler import PassThroughProfiler
from algoperf.pytorch_utils import pytorch_init
from algoperf.pytorch_utils import pytorch_setup
from algoperf.workloads.lm.lm_jax.workload import LmWorkload

# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.set_visible_devices([], 'GPU')

# Environment variables
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disables tensorRT, cuda warnings.
# disable only for deepspeech if it works fine for other workloads
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'


N_GPUS = jax.local_device_count()

print(f"jax.local_devices() = {jax.local_devices()}")
print(f"jax.local_device_count() = {jax.local_device_count()}")

print(f"N_GPUS = {N_GPUS}")

def check_batch(batch):
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

  assert torch.equal(inputs[:,1:], targets[:,:-1])
  

def process_shard(batch):
  inputs, targets = batch['inputs'], batch['targets']
  jax.debug.print("Processing on GPU with inputs:  {shape}", shape=inputs.shape)
  jax.debug.print("inputs {inputs}", inputs=inputs)
  jax.debug.callback(check_batch, batch)
  return inputs, targets

# Apply process_batch across devices, sharding batch across devices
pmap_process = jax.pmap(process_shard, axis_name='batch')


def test_dataloader_jax():
  # Test config.
  rng_seed = 1996
  data_dir = '/fast/najroldi/data/finewebedu'
  split = 'train'
  global_batch_size = 8
  dtype = np.int32
  seq_len = 2048

  local_batch_size = global_batch_size // N_GPUS
  
  workload = LmWorkload()

  data_rng = jax.random.PRNGKey(rng_seed)
  
  input_queue = workload._build_input_queue(
      data_rng=data_rng,
      split=split,
      data_dir=data_dir,
      global_batch_size=global_batch_size)
  
  batch = next(input_queue)
  
  inputs, targets = batch['inputs'], batch['targets']
  print(f"Processing on GPU with inputs: {inputs.shape}")
  
  inputs, targets = pmap_process(batch)
  print(f"Processing on GPU with inputs: {inputs.shape}")
  print(f"Processing on GPU with inputs: {inputs}")

  # inputs, targets = batch['inputs'], batch['targets']
  # print(f"inputs.shape: {inputs.shape}")
  # print(f"inputs[0]: {inputs[0]}")
  # print(f"inputs[1]: {inputs[1]}")
  
  # for device_id in range(2):
  #     # Access the sharded data for each GPU
  #     print(inputs.shape)
      # device_inputs = inputs[device_id]
      # print(f"  GPU {device_id} Inputs: {device_inputs.shape}")
  
  # @jax.pmap
  # def process_batch(batch):    
  #   inputs, targets = batch['inputs'], batch['targets']
  #   print(f"inputs.shape: {inputs.shape}")
    
  #   return inputs, targets
  
  # inputs, targets = batch['inputs'], batch['targets'] #process_batch(batch)
  # print(f"inputs: {inputs[0]}")
  
  

def main():
  test_dataloader_jax()


if __name__ == '__main__':
  main()

