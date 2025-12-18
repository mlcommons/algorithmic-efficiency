"""Benchmark script for PyTorch ImageNet dataloader using shared TFDS pipeline."""

import time

import jax
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Disable TF GPU usage
import tensorflow_datasets as tfds
import torch
import torch.distributed as dist

from algoperf import pytorch_utils
from algoperf.workloads.imagenet_resnet import input_pipeline

# ImageNet constants (same as workload)
TRAIN_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
TRAIN_STDDEV = (0.229 * 255, 0.224 * 255, 0.225 * 255)
CENTER_CROP_SIZE = 224
RESIZE_SIZE = 256
ASPECT_RATIO_RANGE = (0.75, 4.0 / 3.0)
SCALE_RATIO_RANGE = (0.08, 1.0)


def main():
  USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()

  # Initialize DDP process group
  if USE_PYTORCH_DDP:
    torch.cuda.set_device(RANK)
    dist.init_process_group('nccl')

  data_dir = '/home/ak4605/algoperf-data/imagenet/jax'
  global_batch_size = 1024
  num_batches = 100

  if RANK == 0:
    print(f'Creating PyTorch ImageNet dataloader (shared TFDS pipeline)...')
    print(f'Batch size: {global_batch_size}')
    print(f'Num GPUs: {N_GPUS}')
    print(f'USE_PYTORCH_DDP: {USE_PYTORCH_DDP}')

  # Calculate per-device batch size for DDP
  if USE_PYTORCH_DDP:
    batch_size = global_batch_size // N_GPUS
  else:
    batch_size = global_batch_size

  if RANK == 0:
    print(f'Per-device batch size: {batch_size}')

  rng = jax.random.PRNGKey(0)
  ds_builder = tfds.builder('imagenet2012:5.1.0', data_dir=data_dir)

  ds = input_pipeline.create_split(
    split='train',
    dataset_builder=ds_builder,
    rng=rng,
    global_batch_size=batch_size,
    train=True,
    image_size=CENTER_CROP_SIZE,
    resize_size=RESIZE_SIZE,
    mean_rgb=TRAIN_MEAN,
    stddev_rgb=TRAIN_STDDEV,
    cache=False,
    repeat_final_dataset=True,
    aspect_ratio_range=ASPECT_RATIO_RANGE,
    area_range=SCALE_RATIO_RANGE,
    use_mixup=False,
    use_randaug=False,
    image_format='NCHW',
    threadpool_size=48 if USE_PYTORCH_DDP else 48,
  )

  ds_iter = iter(ds)

  def get_batch():
    batch = next(ds_iter)
    inputs = torch.from_numpy(batch['inputs'].numpy()).to(DEVICE)
    targets = torch.from_numpy(batch['targets'].numpy()).to(DEVICE, dtype=torch.long)
    return {'inputs': inputs, 'targets': targets}

  # Warmup
  if RANK == 0:
    print('Warming up...')
  for i in range(5):
    start = time.perf_counter()
    batch = get_batch()
    end = time.perf_counter()
    if RANK == 0:
      print(f'  Warmup batch {i+1}/5: {(end - start)*1000:.2f}ms')

  if RANK == 0:
    print(f"Batch 'inputs' shape: {batch['inputs'].shape}")

  # Synchronize before benchmark
  if USE_PYTORCH_DDP:
    dist.barrier()

  # Benchmark
  if RANK == 0:
    print(f'Benchmarking {num_batches} batches...')
  times = []
  for i in range(num_batches):
    if USE_PYTORCH_DDP:
      dist.barrier()
    start = time.perf_counter()
    batch = get_batch()
    end = time.perf_counter()
    times.append(end - start)
    if RANK == 0 and (i + 1) % 20 == 0:
      print(f'  Batch {i+1}/{num_batches}: {times[-1]*1000:.2f}ms')

  times = np.array(times)
  if RANK == 0:
    print(f'\n=== PyTorch DataLoader Results ===')
    print(f'Mean time per batch: {times.mean()*1000:.2f}ms')
    print(f'Std time per batch: {times.std()*1000:.2f}ms')
    print(f'Min time per batch: {times.min()*1000:.2f}ms')
    print(f'Max time per batch: {times.max()*1000:.2f}ms')
    print(f'Throughput: {global_batch_size / times.mean():.2f} images/sec')

    # Print machine-readable results for the fish script
    print(f'\n=== RESULTS ===')
    print(f'MEAN_MS={times.mean()*1000:.2f}')
    print(f'THROUGHPUT={global_batch_size / times.mean():.2f}')

  if USE_PYTORCH_DDP:
    dist.destroy_process_group()


if __name__ == '__main__':
  main()
