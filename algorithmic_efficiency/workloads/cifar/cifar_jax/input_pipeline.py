"""CIFAR input pipeline.

Forked from Flax example which can be found here:
https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
and adjusted to work for CIFAR10.
"""

import functools
from typing import Dict, Iterator, Tuple

from flax import jax_utils
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

from algorithmic_efficiency import spec
from algorithmic_efficiency.data_utils import shard_and_maybe_pad_np
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax.input_pipeline import \
    normalize_image
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax.input_pipeline import \
    resize


def preprocess_for_train(image: spec.Tensor,
                         rng: spec.RandomState,
                         mean_rgb: Tuple[float, float, float],
                         stddev_rgb: Tuple[float, float, float],
                         crop_size: int,
                         padding_size: int,
                         dtype: tf.DType = tf.float32) -> spec.Tensor:
  """Preprocesses the given image for training.

  Args:
    image: `Tensor` representing an image.
    rng: A per-example, per-step unique RNG seed.
    mean_rgb: A tuple representing the mean of the total training images.
    stddev_rgb: A tuple representing the standard deviation of the
        total training images.
    crop_size: Desired output size of the crop.
    padding_size: An optional padding on each border of the image.
    dtype: data type of the image.

  Returns:
    A preprocessed image `Tensor`.
  """
  rng = tf.random.experimental.stateless_split(rng, 2)
  crop_rng = rng[0, :]
  flip_rng = rng[1, :]

  image_shape = tf.shape(image)
  image = tf.image.resize_with_crop_or_pad(image,
                                           image_shape[0] + padding_size,
                                           image_shape[1] + padding_size)
  image = tf.image.stateless_random_crop(
      image, (crop_size, crop_size, 3), seed=crop_rng)
  image = tf.image.stateless_random_flip_left_right(image, seed=flip_rng)
  image = normalize_image(image, mean_rgb, stddev_rgb, dtype=tf.float32)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image


def preprocess_for_eval(image: spec.Tensor,
                        mean_rgb: Tuple[float, float, float],
                        stddev_rgb: Tuple[float, float, float],
                        image_size: int,
                        dtype: tf.DType = tf.float32) -> spec.Tensor:
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image.
    mean_rgb: A tuple representing the mean of the total training images.
    stddev_rgb: A tuple representing the standard deviation
        of the total training images.
    image_size: A size of the image.
    dtype: data type of the image.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = resize(image, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = normalize_image(image, mean_rgb, stddev_rgb, dtype=dtype)
  return image


def create_split(
    split: str,
    dataset_builder: tfds.core.dataset_builder.DatasetBuilder,
    rng: spec.RandomState,
    global_batch_size: int,
    train: bool,
    mean_rgb: Tuple[float, float, float],
    stddev_rgb: Tuple[float, float, float],
    cache: bool = False,
    repeat_final_dataset: bool = False,
    crop_size: int = 32,
    padding_size: int = 4,
) -> Iterator[Dict[str, spec.Tensor]]:
  """Creates a split from the CIFAR-10 dataset using TensorFlow Datasets."""
  shuffle_rng, preprocess_rng = jax.random.split(rng, 2)

  def preprocess_example(example_index, example):
    dtype = tf.float32
    if train:
      per_step_preprocess_rng = tf.random.experimental.stateless_fold_in(
          tf.cast(preprocess_rng, tf.int64), example_index)
      image = preprocess_for_train(example['image'],
                                   per_step_preprocess_rng,
                                   mean_rgb,
                                   stddev_rgb,
                                   crop_size,
                                   padding_size,
                                   dtype)
    else:
      image = preprocess_for_eval(example['image'],
                                  mean_rgb,
                                  stddev_rgb,
                                  crop_size,
                                  dtype)
    return {'inputs': image, 'targets': example['label']}

  ds = dataset_builder.as_dataset(split=split)
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if cache:
    ds = ds.cache()

  if train or split == 'eval_train':
    ds = ds.repeat()
    ds = ds.shuffle(16 * global_batch_size, seed=shuffle_rng[0])

  # We call ds.enumerate() to get a globally unique per-example, per-step
  # index that we can fold into the RNG seed.
  ds = ds.enumerate()
  ds = ds.map(
      preprocess_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(global_batch_size, drop_remainder=train)

  if repeat_final_dataset:
    ds = ds.repeat()

  ds = ds.prefetch(10)

  return ds


def create_input_iter(
    split: str,
    dataset_builder: tfds.core.dataset_builder.DatasetBuilder,
    rng: spec.RandomState,
    global_batch_size: int,
    mean_rgb: Tuple[float, float, float],
    stddev_rgb: Tuple[float, float, float],
    crop_size: int,
    padding_size: int,
    train: bool,
    cache: bool,
    repeat_final_dataset: bool) -> Iterator[Dict[str, spec.Tensor]]:
  ds = create_split(
      split,
      dataset_builder,
      rng,
      global_batch_size,
      train=train,
      mean_rgb=mean_rgb,
      stddev_rgb=stddev_rgb,
      cache=cache,
      repeat_final_dataset=repeat_final_dataset,
      crop_size=crop_size,
      padding_size=padding_size)
  it = map(
      functools.partial(
          shard_and_maybe_pad_np, global_batch_size=global_batch_size),
      ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it
