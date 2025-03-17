"""
Test data pipaline in JAX and PyTorch.

Instantiate a workload and loops over the input queue.
"""

import jax
import numpy as np
import torch

import algoperf.workloads.lm.lm_jax.workload as lm_jax
# import algoperf.workloads.lm.lm_pytorch.workload as lm_pytorch


data_rng = jax.random.PRNGKey(0)
split = 'train'
data_dir = "/fast/najroldi/data/finewebedu"
seq_len = 2048
global_batch_size = 8
num_batches = 10
repeat_final_dataset = False

# ------------------------------------------------------------------------------
# JAX
# ------------------------------------------------------------------------------

# 1 GPU
workload = lm_jax.LmWorkload()

input_queue = workload._build_input_queue(
    data_rng=data_rng,
    split=split,
    data_dir=data_dir,
    global_batch_size=global_batch_size,
    num_batches=num_batches,
    repeat_final_dataset=repeat_final_dataset)

batch = next(input_queue)
assert type(batch) == dict

assert 'inputs' in batch
assert 'targets' in batch

assert type(batch['inputs']) == np.ndarray
assert type(batch['targets']) == np.ndarray

assert batch['inputs'].dtype == np.int64
assert batch['targets'].dtype == np.int64

assert batch['inputs'].shape == (1, global_batch_size, seq_len)
assert batch['targets'].shape == (1, global_batch_size, seq_len)                      

print(f"JAX devices = {jax.devices()}")
print("1")
