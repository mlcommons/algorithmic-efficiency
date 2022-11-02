"""ImageNet-v2 tf.data input pipeline.

Uses TFDS https://www.tensorflow.org/datasets/catalog/imagenet_v2.
"""

import functools
from typing import Dict, Iterator, Tuple

import tensorflow_datasets as tfds
import torch.distributed as dist

from algorithmic_efficiency import data_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax import \
    input_pipeline

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


def get_imagenet_v2_iter(data_dir: str,
                         global_batch_size: int,
                         mean_rgb: Tuple[float, float, float],
                         stddev_rgb: Tuple[float, float, float],
                         image_size: int,
                         resize_size: int) -> Iterator[Dict[str, spec.Tensor]]:
  """Always caches and repeats indefinitely."""
  if RANK == 0:
    ds = tfds.load(
        'imagenet_v2/matched-frequency:3.0.0',
        split='test',
        data_dir=data_dir,
        decoders={
            'image': tfds.decode.SkipDecoding(),
        })
  if USE_PYTORCH_DDP:
    # If the dataset does not exist, wait for the download.
    dist.barrier()

  if RANK != 0:
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

  ds = ds.map(_decode_example, num_parallel_calls=4)
  ds = ds.batch(global_batch_size)
  shard_pad_fn = functools.partial(
      data_utils.shard_and_maybe_pad_np, global_batch_size=global_batch_size)
  it = map(shard_pad_fn, iter(ds))
  return it
