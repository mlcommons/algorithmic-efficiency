"""ImageNet input pipeline.

Forked from Flax example which can be found here:
https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
"""

from flax import jax_utils
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

IMAGE_SIZE = 224
RESIZE_SIZE = 256
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def _distorted_bounding_box_crop(image_bytes,
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
  shape = tf.io.extract_jpeg_shape(image_bytes)
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
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
  image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  return image


def _resize(image, image_size):
  return tf.image.resize([image], [image_size, image_size],
                         method=tf.image.ResizeMethod.BICUBIC)[0]


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes,
                            rng,
                            image_size,
                            aspect_ratio_range,
                            area_range,
                            resize_size=RESIZE_SIZE):
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


def _decode_and_center_crop(image_bytes, image_size, resize_size):
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


def normalize_image(image, mean_rgb, stddev_rgb):
  image -= tf.constant(mean_rgb, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(stddev_rgb, shape=[1, 1, 3], dtype=image.dtype)
  return image


def preprocess_for_train(image_bytes,
                         rng,
                         mean_rgb,
                         stddev_rgb,
                         aspect_ratio_range,
                         area_range,
                         dtype=tf.float32,
                         image_size=IMAGE_SIZE,
                         resize_size=RESIZE_SIZE):
  """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    rng: a per-example, per-step unique RNG seed.
    dtype: data type of the image.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  crop_rng, flip_rng = tf.random.experimental.stateless_split(rng, 2)

  image = _decode_and_random_crop(image_bytes,
                                  crop_rng,
                                  image_size,
                                  aspect_ratio_range,
                                  area_range,
                                  resize_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.stateless_random_flip_left_right(image, seed=flip_rng)
  image = normalize_image(image, mean_rgb, stddev_rgb)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  return image


def preprocess_for_eval(image_bytes,
                        mean_rgb,
                        stddev_rgb,
                        dtype=tf.float32,
                        image_size=IMAGE_SIZE,
                        resize_size=RESIZE_SIZE):
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
                 num_batches=None,
                 aspect_ratio_range=(0.75, 4.0 / 3.0),
                 area_range=(0.08, 1.0)):
  """Creates a split from the ImageNet dataset using TensorFlow Datasets."""
  if split == 'eval_train':
    split = 'train'

  shuffle_rng, preprocess_rng = jax.random.split(rng, 2)

  def decode_example(example):
    dtype = tf.float32
    if train:
      # We call ds.enumerate() to get a globally unique per-example, per-step
      # index that we can fold into the RNG seed.
      (example_index, example) = example
      per_step_preprocess_rng = tf.random.experimental.stateless_fold_in(
          tf.cast(preprocess_rng, tf.int64), example_index)
      image = preprocess_for_train(example['image'],
                                   per_step_preprocess_rng,
                                   example_index,
                                   mean_rgb,
                                   stddev_rgb,
                                   aspect_ratio_range,
                                   area_range,
                                   dtype,
                                   image_size,
                                   resize_size)
    else:
      image = preprocess_for_eval(example['image'],
                                  mean_rgb,
                                  stddev_rgb,
                                  dtype,
                                  image_size,
                                  resize_size)
    return {'inputs': image, 'targets': example['label']}

  ds = dataset_builder.as_dataset(
      split=split, decoders={
          'image': tfds.decode.SkipDecoding(),
      })
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if cache:
    ds = ds.cache()

  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * global_batch_size, seed=shuffle_rng[0])

  ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(global_batch_size, drop_remainder=True)

  if num_batches is not None:
    ds = ds.take(num_batches)

  if repeat_final_dataset:
    ds = ds.repeat()

  ds = ds.prefetch(10)

  return ds


def shard_numpy_ds(xs):
  """Prepare tf data for JAX

  Convert an input batch from tf Tensors to numpy arrays and reshape it to be
  sharded across devices.
  """
  local_device_count = jax.local_device_count()

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def create_input_iter(split,
                      dataset_builder,
                      rng,
                      global_batch_size,
                      mean_rgb,
                      stddev_rgb,
                      image_size,
                      resize_size,
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
      dtype=tf.float32,
      image_size=image_size,
      resize_size=resize_size,
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
