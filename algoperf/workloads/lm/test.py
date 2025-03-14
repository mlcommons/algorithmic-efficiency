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

next(input_queue)
