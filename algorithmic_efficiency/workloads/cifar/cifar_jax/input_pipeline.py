"""CIFAR input pipeline.

Forked from Flax example which can be found here:
https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
and adjusted to work for CIFAR10.
"""

from flax import jax_utils
import jax
import tensorflow as tf

from algorithmic_efficiency.data_utils import shard_numpy_ds

IMAGE_SIZE = 32
MEAN_RGB = [0.49139968 * 255, 0.48215827 * 255, 0.44653124 * 255]
STDDEV_RGB = [0.24703233 * 255, 0.24348505 * 255, 0.26158768 * 255]


def _distorted_bounding_box_crop(image,
                                 rng,
                                 bbox,
                                 min_object_covered=0.1,
                                 aspect_ratio_range=(0.75, 1.33),
                                 area_range=(0.05, 1.0),
                                 max_attempts=100):
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
  shape = tf.shape(image)
  sample_distorted_bounding_box = tf.image.stateless_sample_distorted_bounding_box(  # pylint: disable=line-too-long
      shape,
      seed=rng,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  image = tf.image.crop_to_bounding_box(image,
                                        offset_y,
                                        offset_x,
                                        target_height,
                                        target_width)
  return image


def _resize(image, image_size):
  return tf.image.resize([image], [image_size, image_size],
                         method=tf.image.ResizeMethod.BICUBIC)[0]


def _random_crop(image, rng, image_size, aspect_ratio_range, area_range):
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

  return _resize(image_cropped, image_size)


def normalize_image(image, mean_rgb, stddev_rgb):
  image -= tf.constant(mean_rgb, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(stddev_rgb, shape=[1, 1, 3], dtype=image.dtype)
  return image


def preprocess_for_train(image,
                         rng,
                         mean_rgb,
                         stddev_rgb,
                         aspect_ratio_range,
                         area_range,
                         dtype=tf.float32,
                         image_size=IMAGE_SIZE):
  """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    rng: a per-example, per-step unique RNG seed.
    dtype: data type of the image.
    image_size: image size.

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


def preprocess_for_eval(image,
                        mean_rgb,
                        stddev_rgb,
                        dtype=tf.float32,
                        image_size=IMAGE_SIZE):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _resize(image, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = normalize_image(image, mean_rgb, stddev_rgb)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image


def create_split(split,
                 dataset_builder,
                 rng,
                 global_batch_size,
                 train,
                 image_size,
                 mean_rgb,
                 stddev_rgb,
                 cache=False,
                 repeat_final_dataset=False,
                 num_batches=None,
                 aspect_ratio_range=(0.75, 4.0 / 3.0),
                 area_range=(0.08, 1.0)):
  """Creates a split from the ImageNet dataset using TensorFlow Datasets."""
  if split == 'eval_train':
    split = 'train'
  elif split == 'validation':
    split = 'test'

  shuffle_rng, preprocess_rng = jax.random.split(rng, 2)

  def preprocess_example(example):
    dtype = tf.float32
    # We call ds.enumerate() to get a globally unique per-example, per-step
    # index that we can fold into the RNG seed.
    (example_index, example) = example
    if train:
      per_step_preprocess_rng = tf.random.experimental.stateless_fold_in(
          tf.cast(preprocess_rng, tf.int64), example_index)
      image = preprocess_for_train(example['image'],
                                   per_step_preprocess_rng,
                                   mean_rgb,
                                   stddev_rgb,
                                   aspect_ratio_range,
                                   area_range,
                                   dtype,
                                   image_size)
    else:
      image = preprocess_for_eval(example['image'],
                                  mean_rgb,
                                  stddev_rgb,
                                  dtype,
                                  image_size)
    return {'inputs': image, 'targets': example['label']}

  ds = dataset_builder.as_dataset(split=split)
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if cache:
    ds = ds.cache()

  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * global_batch_size, seed=shuffle_rng[0])

  ds = ds.enumerate().map(
      lambda i,
      ex: preprocess_example((i, ex)),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(global_batch_size, drop_remainder=train)

  if num_batches is not None:
    ds = ds.take(num_batches)

  if repeat_final_dataset:
    ds = ds.repeat()

  ds = ds.prefetch(10)

  return ds


def create_input_iter(split,
                      dataset_builder,
                      rng,
                      global_batch_size,
                      mean_rgb,
                      stddev_rgb,
                      image_size,
                      aspect_ratio_range,
                      area_range,
                      train,
                      cache,
                      repeat_final_dataset,
                      num_batches):
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
      num_batches=num_batches,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range)
  it = map(shard_numpy_ds, ds)

  # Note(Dan S): On a Nvidia 2080 Ti GPU, this increased GPU utilization by 10%.
  it = jax_utils.prefetch_to_device(it, 2)

  return it
