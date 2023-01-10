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


def _distorted_bounding_box_crop(image: spec.Tensor,
                                 rng: spec.RandomState,
                                 bbox: spec.Tensor,
                                 min_object_covered: float = 0.1,
                                 aspect_ratio_range: Tuple[float,
                                                           float] = (0.75,
                                                                     1.33),
                                 area_range: Tuple[float, float] = (0.05, 1.0),
                                 max_attempts: int = 100) -> spec.Tensor:
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: `Tensor` of image data.
    rng: A per-example, per-step unique RNG seed.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.

  Returns:
    Cropped image `Tensor`.
  """
  shape = tf.shape(image)
  bbox_begin, bbox_size, _ = tf.image.stateless_sample_distorted_bounding_box(
      shape,
      seed=rng,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  image = tf.image.crop_to_bounding_box(image,
                                        offset_y,
                                        offset_x,
                                        target_height,
                                        target_width)
  return image


def _random_crop(image: spec.Tensor,
                 rng: spec.RandomState,
                 image_size: int,
                 aspect_ratio_range: Tuple[float, float],
                 area_range: Tuple[float, float]) -> spec.Tensor:
  """Make a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image_cropped = _distorted_bounding_box_crop(
      image,
      rng,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=10)
  return resize(image_cropped, image_size)


def preprocess_for_train(image: spec.Tensor,
                         rng: spec.RandomState,
                         mean_rgb: Tuple[float, float, float],
                         stddev_rgb: Tuple[float, float, float],
                         aspect_ratio_range: Tuple[float, float],
                         area_range: Tuple[float, float],
                         image_size: int,
                         dtype: tf.DType = tf.float32) -> spec.Tensor:
  """Preprocesses the given image for training.

  Args:
    image: `Tensor` representing an image.
    rng: A per-example, per-step unique RNG seed.
    mean_rgb: A tuple representing the mean of the total training images.
    stddev_rgb: A tuple representing the standard deviation of the
        total training images.
    aspect_ratio_range: An optional tuple of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional tuple of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    image_size: A size of the image.
    dtype: data type of the image.

  Returns:
    A preprocessed image `Tensor`.
  """
  rng = tf.random.experimental.stateless_split(rng, 2)
  crop_rng = rng[0, :]
  flip_rng = rng[1, :]

  image = _random_crop(image,
                       crop_rng,
                       image_size,
                       aspect_ratio_range,
                       area_range)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.stateless_random_flip_left_right(image, seed=flip_rng)
  image = normalize_image(image, mean_rgb, stddev_rgb)
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
  image = normalize_image(image, mean_rgb, stddev_rgb)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image


def create_split(
    split: str,
    dataset_builder: tfds.core.dataset_builder.DatasetBuilder,
    rng: spec.RandomState,
    global_batch_size: int,
    train: bool,
    image_size: int,
    mean_rgb: Tuple[float, float, float],
    stddev_rgb: Tuple[float, float, float],
    cache: bool = False,
    repeat_final_dataset: bool = False,
    aspect_ratio_range: Tuple[float, float] = (0.75, 4.0 / 3.0),
    area_range: Tuple[float, float] = (0.08, 1.0)
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
                                   aspect_ratio_range,
                                   area_range,
                                   image_size,
                                   dtype)
    else:
      image = preprocess_for_eval(example['image'],
                                  mean_rgb,
                                  stddev_rgb,
                                  image_size,
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
    image_size: int,
    aspect_ratio_range: Tuple[float, float],
    area_range: Tuple[float, float],
    train: bool,
    cache: bool,
    repeat_final_dataset: bool) -> Iterator[Dict[str, spec.Tensor]]:
  ds = create_split(
      split,
      dataset_builder,
      rng,
      global_batch_size,
      train=train,
      image_size=image_size,
      mean_rgb=mean_rgb,
      stddev_rgb=stddev_rgb,
      cache=cache,
      repeat_final_dataset=repeat_final_dataset,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range)
  it = map(
      functools.partial(
          shard_and_maybe_pad_np, global_batch_size=global_batch_size),
      ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it
