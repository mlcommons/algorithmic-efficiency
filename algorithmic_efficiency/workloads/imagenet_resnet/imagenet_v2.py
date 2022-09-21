"""ImageNet-v2 tf.data input pipeline.

Uses TFDS https://www.tensorflow.org/datasets/catalog/imagenet_v2.
"""
from flax import jax_utils
import itertools
import jax
import numpy as np
import tensorflow_datasets as tfds

from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax import \
    input_pipeline


def _shard(x):
  return x.reshape((jax.local_device_count(), -1, *x.shape[1:]))


def shard_and_maybe_pad_batch(desired_batch_size, tf_batch):
  batch_axis = 0
  batch = jax.tree_map(lambda x: x._numpy(), tf_batch)
  batch_size = batch['inputs'].shape[batch_axis]
  batch['weights'] = np.ones(batch_size, dtype=np.float32)

  batch_pad = desired_batch_size - batch_size
  # Most batches will not need padding so we quickly return to avoid slowdown.
  if batch_pad == 0:
    return jax.tree_map(_shard, batch)

  def zero_pad(x):
    padding = [(0, 0)] * x.ndim
    padding[batch_axis] = (0, batch_pad)
    padded = np.pad(x, padding, mode='constant', constant_values=0.0)
    return _shard(padded)

  return jax.tree_map(zero_pad, batch)


def get_imagenet_v2_iter(data_dir, global_batch_size, mean_rgb, stddev_rgb):
  """Always caches and repeats indefinitely."""
  ds = tfds.load(
      'imagenet_v2/matched-frequency:3.0.0',
      split='test',
      data_dir=data_dir,
      decoders={
          'image': tfds.decode.SkipDecoding(),
      })

  def _decode_example(example):
    image = input_pipeline.preprocess_for_eval(
        example['image'], mean_rgb, stddev_rgb)
    return {
        'inputs': image,
        'targets': example['label'],
    }

  ds = ds.map(_decode_example, num_parallel_calls=16)
  ds = ds.batch(global_batch_size)
  it = map(shard_and_maybe_pad_batch, iter(ds))
  return itertools.cycle(it)
