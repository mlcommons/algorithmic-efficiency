"""ImageNet-v2 tf.data input pipeline.

Uses TFDS https://www.tensorflow.org/datasets/catalog/imagenet_v2.
"""

import functools
from typing import Dict, Iterator, Tuple

import jax
import numpy as np
import tensorflow_datasets as tfds
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax import \
    input_pipeline


def _shard(x: spec.Tensor) -> spec.Tensor:
  # If we install the CPU version of a framework it may not return the correct
  # number of GPUs.
  num_devices = max(torch.cuda.device_count(), jax.local_device_count())
  return x.reshape((num_devices, -1, *x.shape[1:]))


def shard_and_maybe_pad_batch(desired_batch_size: int,
                              shard_batch: bool,
                              tf_batch: Dict[str, float]) -> Dict[str, float]:
  batch_axis = 0
  batch = jax.tree_map(lambda x: x._numpy(), tf_batch)
  batch_size = batch['inputs'].shape[batch_axis]
  batch['weights'] = np.ones(batch_size, dtype=np.float32)

  batch_pad = desired_batch_size - batch_size
  # Most batches will not need padding; we quickly return to avoid slowdown.
  if batch_pad == 0:
    if shard_batch:
      return jax.tree_map(_shard, batch)
    else:
      return batch

  def zero_pad(x: spec.Tensor) -> spec.Tensor:
    padding = [(0, 0)] * x.ndim
    padding[batch_axis] = (0, batch_pad)
    padded = np.pad(x, padding, mode='constant', constant_values=0.0)
    if shard_batch:
      return _shard(padded)
    else:
      return padded

  return jax.tree_map(zero_pad, batch)


def get_imagenet_v2_iter(data_dir: str,
                         global_batch_size: int,
                         shard_batch: bool,
                         mean_rgb: Tuple[float, float, float],
                         stddev_rgb: Tuple[float, float, float],
                         image_size: int,
                         resize_size: int) -> Iterator[Dict[str, spec.Tensor]]:
  """Always caches and repeats indefinitely."""
  ds = tfds.load(
      'imagenet_v2/matched-frequency:3.0.0',
      split='test',
      data_dir=data_dir,
      decoders={
          'image': tfds.decode.SkipDecoding(),
      })

  def _decode_example(example: Dict[str, float]) -> Dict[str, float]:
    image = input_pipeline.preprocess_for_eval(example['image'],
                                               mean_rgb,
                                               stddev_rgb,
                                               image_size,
                                               resize_size)
    return {'inputs': image, 'targets': example['label']}

  ds = ds.map(_decode_example, num_parallel_calls=16)
  ds = ds.batch(global_batch_size)
  shard_pad_fn = functools.partial(shard_and_maybe_pad_batch,
                                   global_batch_size,
                                   shard_batch)
  it = map(shard_pad_fn, iter(ds))
  return it
