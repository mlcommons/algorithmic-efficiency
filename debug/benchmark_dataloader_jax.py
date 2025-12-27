"""Benchmark script for JAX ImageNet dataloader."""

import time

import jax
import numpy as np
import tensorflow_datasets as tfds

from algoperf.workloads.imagenet_resnet import input_pipeline

# ImageNet constants (same as workload)
TRAIN_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
TRAIN_STDDEV = (0.229 * 255, 0.224 * 255, 0.225 * 255)
CENTER_CROP_SIZE = 224
RESIZE_SIZE = 256
ASPECT_RATIO_RANGE = (0.75, 4.0 / 3.0)
SCALE_RATIO_RANGE = (0.08, 1.0)


def main():
  data_dir = '/home/ak4605/data/imagenet/jax'
  global_batch_size = 1024
  num_batches = 100

  rng = jax.random.PRNGKey(0)
  ds_builder = tfds.builder('imagenet2012:5.1.0', data_dir=data_dir)

  print('Creating JAX ImageNet dataloader...')
  print(f'Batch size: {global_batch_size}')
  print(f'Num devices: {jax.local_device_count()}')

  ds = input_pipeline.create_split(
    split='train',
    dataset_builder=ds_builder,
    rng=rng,
    global_batch_size=global_batch_size,
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
    image_format='NHWC',
  )

  ds_iter = iter(ds)

  # Warmup
  print('Warming up...')
  for i in range(5):
    start = time.perf_counter()
    batch = next(ds_iter)
    end = time.perf_counter()
    print(f'  Warmup batch {i + 1}/5: {(end - start) * 1000:.2f}ms')

  print(f"Batch 'inputs' shape: {batch['inputs'].shape}")

  # Benchmark
  print(f'Benchmarking {num_batches} batches...')
  times = []
  for i in range(num_batches):
    start = time.perf_counter()
    batch = next(ds_iter)
    # Force sync by accessing data
    _ = np.asarray(batch['inputs'][0, 0, 0, 0])
    end = time.perf_counter()
    times.append(end - start)
    if (i + 1) % 20 == 0:
      print(f'  Batch {i + 1}/{num_batches}: {times[-1] * 1000:.2f}ms')

  times = np.array(times)
  print('\n=== JAX DataLoader Results ===')
  print(f'Mean time per batch: {times.mean() * 1000:.2f}ms')
  print(f'Std time per batch: {times.std() * 1000:.2f}ms')
  print(f'Min time per batch: {times.min() * 1000:.2f}ms')
  print(f'Max time per batch: {times.max() * 1000:.2f}ms')
  print(f'Throughput: {global_batch_size / times.mean():.2f} images/sec')

  # Print machine-readable results for the fish script
  print('\n=== RESULTS ===')
  print(f'MEAN_MS={times.mean() * 1000:.2f}')
  print(f'THROUGHPUT={global_batch_size / times.mean():.2f}')


if __name__ == '__main__':
  main()
