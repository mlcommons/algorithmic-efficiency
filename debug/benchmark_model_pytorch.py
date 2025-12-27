"""Benchmark script for PyTorch ImageNet ResNet50 model (forward + backward)."""

import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from algoperf import pytorch_utils
from algoperf.workloads.imagenet_resnet.imagenet_pytorch.models import resnet50

# Training config
BATCH_SIZE = 1024
IMAGE_SIZE = 224
NUM_CLASSES = 1000
NUM_WARMUP = 10
NUM_BENCHMARK = 100


def main():
  USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()

  # Initialize DDP process group
  if USE_PYTORCH_DDP:
    torch.cuda.set_device(RANK)
    dist.init_process_group('nccl')

  if RANK == 0:
    print('=== PyTorch ResNet50 Model Benchmark ===')
    print(f'Global batch size: {BATCH_SIZE}')
    print(f'Image size: {IMAGE_SIZE}')
    print(f'Num GPUs: {N_GPUS}')
    print(f'USE_PYTORCH_DDP: {USE_PYTORCH_DDP}')
    print(f'Device: {DEVICE}')

  # Calculate per-device batch size
  if USE_PYTORCH_DDP:
    per_device_batch_size = BATCH_SIZE // N_GPUS
  else:
    per_device_batch_size = BATCH_SIZE

  if RANK == 0:
    print(f'Per-device batch size: {per_device_batch_size}')

  # Initialize model
  if RANK == 0:
    print('\nInitializing model...')

  torch.manual_seed(0)
  model = resnet50(act_fnc=torch.nn.ReLU(inplace=True))
  model.to(DEVICE)

  if USE_PYTORCH_DDP:
    model = DDP(model, device_ids=[RANK], output_device=RANK)

  param_count = sum(p.numel() for p in model.parameters())
  if RANK == 0:
    print(f'Model initialized. Param count: {param_count:,}')

  # Compile model with torch.compile for optimized performance
  if RANK == 0:
    print('Compiling model with torch.compile...')
  model = torch.compile(model)

  # Generate random data (NCHW format for PyTorch)
  if RANK == 0:
    print('Generating random data...')

  torch.manual_seed(42 + RANK)  # Different data per rank
  inputs = torch.randn(
    per_device_batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE
  )
  targets = torch.randint(
    0, NUM_CLASSES, (per_device_batch_size,), device=DEVICE
  )

  # Warmup forward pass (includes torch.compile compilation)
  if RANK == 0:
    print(f'\nWarming up forward pass ({NUM_WARMUP} iterations)...')

  model.eval()
  with torch.no_grad():
    for i in range(NUM_WARMUP):
      if USE_PYTORCH_DDP:
        dist.barrier()
      start = time.perf_counter()
      logits = model(inputs)
      torch.cuda.synchronize()
      end = time.perf_counter()
      if RANK == 0:
        print(f'  Warmup {i + 1}/{NUM_WARMUP}: {(end - start) * 1000:.2f}ms')

  # Benchmark forward pass
  if RANK == 0:
    print(f'\nBenchmarking forward pass ({NUM_BENCHMARK} iterations)...')

  forward_times = []
  with torch.no_grad():
    for i in range(NUM_BENCHMARK):
      if USE_PYTORCH_DDP:
        dist.barrier()
      start = time.perf_counter()
      logits = model(inputs)
      torch.cuda.synchronize()
      end = time.perf_counter()
      forward_times.append(end - start)
      if RANK == 0 and (i + 1) % 20 == 0:
        print(
          f'  Batch {i + 1}/{NUM_BENCHMARK}: {forward_times[-1] * 1000:.2f}ms'
        )

  forward_times = np.array(forward_times)
  if RANK == 0:
    print('\n--- Forward Pass Results ---')
    print(f'Mean: {forward_times.mean() * 1000:.2f}ms')
    print(f'Std: {forward_times.std() * 1000:.2f}ms')
    print(f'Min: {forward_times.min() * 1000:.2f}ms')
    print(f'Max: {forward_times.max() * 1000:.2f}ms')
    print(f'Throughput: {BATCH_SIZE / forward_times.mean():.2f} images/sec')

  # Warmup forward + backward pass (includes torch.compile compilation for backward)
  if RANK == 0:
    print(f'\nWarming up forward+backward pass ({NUM_WARMUP} iterations)...')

  model.train()
  for i in range(NUM_WARMUP):
    if USE_PYTORCH_DDP:
      dist.barrier()
    start = time.perf_counter()
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    torch.cuda.synchronize()
    end = time.perf_counter()
    model.zero_grad()
    if RANK == 0:
      print(f'  Warmup {i + 1}/{NUM_WARMUP}: {(end - start) * 1000:.2f}ms')

  # Benchmark forward + backward pass
  if RANK == 0:
    print(
      f'\nBenchmarking forward+backward pass ({NUM_BENCHMARK} iterations)...'
    )

  train_times = []
  for i in range(NUM_BENCHMARK):
    if USE_PYTORCH_DDP:
      dist.barrier()
    start = time.perf_counter()
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    torch.cuda.synchronize()
    end = time.perf_counter()
    train_times.append(end - start)
    model.zero_grad()
    if RANK == 0 and (i + 1) % 20 == 0:
      print(f'  Batch {i + 1}/{NUM_BENCHMARK}: {train_times[-1] * 1000:.2f}ms')

  train_times = np.array(train_times)
  if RANK == 0:
    print('\n--- Forward+Backward Pass Results ---')
    print(f'Mean: {train_times.mean() * 1000:.2f}ms')
    print(f'Std: {train_times.std() * 1000:.2f}ms')
    print(f'Min: {train_times.min() * 1000:.2f}ms')
    print(f'Max: {train_times.max() * 1000:.2f}ms')
    print(f'Throughput: {BATCH_SIZE / train_times.mean():.2f} images/sec')

    # Print machine-readable results for the fish script
    print('\n=== RESULTS ===')
    print(f'FORWARD_MEAN_MS={forward_times.mean() * 1000:.2f}')
    print(f'FORWARD_THROUGHPUT={BATCH_SIZE / forward_times.mean():.2f}')
    print(f'TRAIN_MEAN_MS={train_times.mean() * 1000:.2f}')
    print(f'TRAIN_THROUGHPUT={BATCH_SIZE / train_times.mean():.2f}')

  if USE_PYTORCH_DDP:
    dist.destroy_process_group()


if __name__ == '__main__':
  main()
