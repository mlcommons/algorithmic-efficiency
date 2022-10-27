"""ImageNet-v2 tf.data input pipeline.

Uses TFDS https://www.tensorflow.org/datasets/catalog/imagenet_v2.
"""
import functools

import tensorflow_datasets as tfds

from algorithmic_efficiency import data_utils
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax import \
    input_pipeline


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
    image = input_pipeline.preprocess_for_eval(example['image'],
                                               mean_rgb,
                                               stddev_rgb)
    return {
        'inputs': image,
        'targets': example['label'],
    }

  ds = ds.map(_decode_example, num_parallel_calls=16)
  ds = ds.batch(global_batch_size)
  shard_pad_fn = functools.partial(
      data_utils.shard_numpy_ds, global_batch_size=global_batch_size)
  it = map(shard_pad_fn, iter(ds))
  return it
