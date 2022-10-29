"""ImageNet input pipeline.

Forked from Flax example which can be found here:
https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py.
"""

import functools
from typing import Dict, Iterator, Tuple

from flax import jax_utils
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from algorithmic_efficiency import data_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax import \
    randaugment


def _distorted_bounding_box_crop(image_bytes: spec.Tensor,
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
    image_bytes: `Tensor` of binary image data.
    rng: a per-example, per-step unique RNG seed.
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
    cropped image `Tensor`
  """
  shape = tf.io.extract_jpeg_shape(image_bytes)
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
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
  image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  return image


def _resize(image: spec.Tensor, image_size: int) -> spec.Tensor:
  return tf.image.resize([image], [image_size, image_size],
                         method=tf.image.ResizeMethod.BICUBIC)[0]


def _at_least_x_are_equal(a: spec.Tensor, b: spec.Tensor, x: float) -> bool:
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes: spec.Tensor,
                            rng: spec.RandomState,
                            image_size: int,
                            aspect_ratio_range: Tuple[float, float],
                            area_range: Tuple[float, float],
                            resize_size: int) -> spec.Tensor:
  """Make a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = _distorted_bounding_box_crop(
      image_bytes,
      rng,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=10)
  original_shape = tf.io.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(
      bad,
      lambda: _decode_and_center_crop(image_bytes, image_size, resize_size),
      lambda: _resize(image, image_size))

  return image


def _decode_and_center_crop(image_bytes: spec.Tensor,
                            image_size: int,
                            resize_size: int) -> spec.Tensor:
  """Crops to center of image with padding then scales image_size."""
  shape = tf.io.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / resize_size) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([
      offset_height,
      offset_width,
      padded_center_crop_size,
      padded_center_crop_size
  ])
  image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = _resize(image, image_size)

  return image


def normalize_image(image: spec.Tensor,
                    mean_rgb: Tuple[float, float, float],
                    stddev_rgb: Tuple[float, float, float]) -> spec.Tensor:
  image -= tf.constant(mean_rgb, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(stddev_rgb, shape=[1, 1, 3], dtype=image.dtype)
  return image


def preprocess_for_train(image_bytes: spec.Tensor,
                         rng: spec.RandomState,
                         mean_rgb: Tuple[float, float, float],
                         stddev_rgb: Tuple[float, float, float],
                         aspect_ratio_range: Tuple[float, float],
                         area_range: Tuple[float, float],
                         image_size: int,
                         resize_size: int,
                         dtype: tf.DType = tf.float32,
                         use_randaug: bool = False,
                         randaug_num_layers: int = 2,
                         randaug_magnitude: int = 10) -> spec.Tensor:
  """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    rng: a per-example, per-step unique RNG seed.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  rngs = tf.random.experimental.stateless_split(rng, 3)

  image = _decode_and_random_crop(image_bytes,
                                  rngs[0],
                                  image_size,
                                  aspect_ratio_range,
                                  area_range,
                                  resize_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.stateless_random_flip_left_right(image, seed=rngs[1])

  if use_randaug:
    image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
    image = randaugment.distort_image_with_randaugment(image,
                                                       randaug_num_layers,
                                                       randaug_magnitude,
                                                       rngs[2])
  image = tf.cast(image, tf.float32)
  image = normalize_image(image, mean_rgb, stddev_rgb)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image


def preprocess_for_eval(image_bytes: spec.Tensor,
                        mean_rgb: Tuple[float, float, float],
                        stddev_rgb: Tuple[float, float, float],
                        image_size: int,
                        resize_size: int,
                        dtype: tf.DType = tf.float32) -> spec.Tensor:
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image_bytes, image_size, resize_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = normalize_image(image, mean_rgb, stddev_rgb)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image


# Modified from
# github.com/google/init2winit/blob/master/init2winit/dataset_lib/ (cont. below)
# image_preprocessing.py.
def mixup_tf(key: spec.RandomState,
             inputs: spec.Tensor,
             targets: spec.Tensor,
             alpha: float = 0.2) -> Tuple[spec.Tensor, spec.Tensor]:
  """Perform mixup https://arxiv.org/abs/1710.09412.

  NOTE: Code taken from https://github.com/google/big_vision with variables
  renamed to match `mixup` in this file and logic to synchronize globally.

  Args:
    key: The random key to use.
    inputs: inputs to mix.
    targets: targets to mix.
    alpha: the beta/dirichlet concentration parameter, typically 0.1 or 0.2.

  Returns:
    Mixed inputs and targets.
  """
  # Transform to one-hot targets.
  targets = tf.one_hot(targets, 1000)
  # Compute weight for convex combination by sampling from Beta distribution.
  beta_dist = tfp.distributions.Beta(alpha, alpha)
  weight = beta_dist.sample(seed=tf.cast(key[0], tf.int32))
  # Return convex combination of original and shifted inputs and targets.
  inputs = weight * inputs + (1.0 - weight) * tf.roll(inputs, 1, axis=0)
  targets = weight * targets + (1.0 - weight) * tf.roll(targets, 1, axis=0)
  return inputs, targets


def create_split(split,
                 dataset_builder,
                 rng,
                 global_batch_size,
                 train,
                 image_size,
                 resize_size,
                 mean_rgb,
                 stddev_rgb,
                 cache=False,
                 repeat_final_dataset=False,
                 aspect_ratio_range=(0.75, 4.0 / 3.0),
                 area_range=(0.08, 1.0),
                 use_mixup=False,
                 mixup_alpha=0.1,
                 use_randaug=False,
                 randaug_num_layers=2,
                 randaug_magnitude=10) -> Iterator[Dict[str, spec.Tensor]]:
  """Creates a split from the ImageNet dataset using TensorFlow Datasets."""

  shuffle_rng, preprocess_rng, mixup_rng = jax.random.split(rng, 3)

  def decode_example(example_index, example):
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
                                   resize_size,
                                   dtype,
                                   use_randaug,
                                   randaug_num_layers,
                                   randaug_magnitude)
    else:
      image = preprocess_for_eval(example['image'],
                                  mean_rgb,
                                  stddev_rgb,
                                  image_size,
                                  resize_size,
                                  dtype)
    return {'inputs': image, 'targets': example['label']}

  ds = dataset_builder.as_dataset(
      split=split, decoders={
          'image': tfds.decode.SkipDecoding(),
      })
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if cache:
    ds = ds.cache()

  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * global_batch_size, seed=shuffle_rng[0])

  # We call ds.enumerate() to get a globally unique per-example, per-step
  # index that we can fold into the RNG seed.
  ds = ds.enumerate()
  ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(global_batch_size, drop_remainder=train)

  if use_mixup:
    if train:

      def mixup_batch(batch_index, batch):
        per_batch_mixup_rng = tf.random.experimental.stateless_fold_in(
            mixup_rng, batch_index)
        (inputs, targets) = mixup_tf(
            per_batch_mixup_rng,
            batch['inputs'],
            batch['targets'],
            alpha=mixup_alpha)
        batch['inputs'] = inputs
        batch['targets'] = targets
        return batch

      ds = ds.enumerate().map(
          mixup_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
      raise ValueError('Mixup can only be used for the training split.')

  if repeat_final_dataset:
    ds = ds.repeat()

  ds = ds.prefetch(10)

  return ds


def create_input_iter(split: str,
                      dataset_builder: tfds.core.dataset_builder.DatasetBuilder,
                      rng: spec.RandomState,
                      global_batch_size: int,
                      mean_rgb: Tuple[float, float, float],
                      stddev_rgb: Tuple[float, float, float],
                      image_size: int,
                      resize_size: int,
                      aspect_ratio_range: Tuple[float, float],
                      area_range: Tuple[float, float],
                      train: bool,
                      cache: bool,
                      repeat_final_dataset: bool,
                      use_mixup: bool,
                      mixup_alpha: float,
                      use_randaug: bool) -> Iterator[Dict[str, spec.Tensor]]:
  ds = create_split(
      split,
      dataset_builder,
      rng,
      global_batch_size,
      train=train,
      image_size=image_size,
      resize_size=resize_size,
      mean_rgb=mean_rgb,
      stddev_rgb=stddev_rgb,
      cache=cache,
      repeat_final_dataset=repeat_final_dataset,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      use_mixup=use_mixup,
      mixup_alpha=mixup_alpha,
      use_randaug=use_randaug)
  it = map(
      functools.partial(
          data_utils.shard_and_maybe_pad_np, global_batch_size=global_batch_size),
      ds)

  # Note(Dan S): On a Nvidia 2080 Ti GPU, this increased GPU utilization by 10%.
  it = jax_utils.prefetch_to_device(it, 2)

  return iter(it)
