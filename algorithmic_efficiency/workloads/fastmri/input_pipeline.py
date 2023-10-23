"""FastMRI knee singlecoil input pipeline."""

import datetime
import functools
import glob
import os

import h5py
import jax
import tensorflow as tf

from algorithmic_efficiency import data_utils

_TRAIN_DIR = 'knee_singlecoil_train'
_VAL_DIR = 'knee_singlecoil_val'
_EVAL_SEED = 0


def _process_example(kspace,
                     kspace_shape,
                     target,
                     target_shape,
                     volume_max,
                     seed):
  """Generate a single example (slice from mri image).

  Args:
    kspace: raw mri data.
    kspace_shape: shape of kspace. We pass this in because it is not constant.
    target: target image.
    target_shape: shape of target.
    volume_max: max value over the entire volume that the example slice was
      originally derived from.
    seed: seed for stateless randomness.
  Returns:
    dictionary of processed image/target.
  """

  # sample_mask
  num_cols = kspace_shape[1]
  num_cols_float = tf.cast(num_cols, dtype=tf.float32)

  # choose_acceleration
  center_fraction = tf.convert_to_tensor(0.08, dtype=tf.float32)
  acceleration = tf.convert_to_tensor(4.0, dtype=tf.float32)

  num_low_frequencies = tf.cast(
      num_cols_float * center_fraction, dtype=tf.int32)

  # calculate_center_mask
  mask = tf.zeros(num_cols, dtype=tf.float32)
  pad = (num_cols - num_low_frequencies + 1) // 2
  mask = tf.tensor_scatter_nd_update(
      mask,
      tf.reshape(tf.range(pad, pad + num_low_frequencies), (-1, 1)),
      tf.ones(num_low_frequencies))

  # reshape_mask
  center_mask = tf.reshape(mask, (1, num_cols))

  # calculate_acceleration_mask
  num_low_frequencies_float = tf.cast(num_low_frequencies, dtype=tf.float32)
  prob = (num_cols_float / acceleration - num_low_frequencies_float) / (
      num_cols_float - num_low_frequencies_float)

  mask = tf.cast(
      tf.random.stateless_uniform((num_cols,), seed) < prob, dtype=tf.float32)
  acceleration_mask = tf.reshape(mask, (1, num_cols))

  mask = tf.math.maximum(center_mask, acceleration_mask)
  mask = tf.cast(mask, dtype=tf.complex64)

  # apply_mask
  masked_kspace = kspace * mask + 0.0

  # ifft2c
  shifted_kspace = tf.signal.ifftshift(masked_kspace, axes=(0, 1))
  shifted_image = tf.signal.ifft2d(shifted_kspace)
  image = tf.signal.fftshift(shifted_image, axes=(0, 1))
  scaling_norm = tf.cast(
      tf.math.sqrt(
          tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), 'float32')),
      kspace.dtype)
  image = image * scaling_norm
  image = tf.stack((tf.math.real(image), tf.math.imag(image)), axis=-1)

  # complex_center_crop
  w_from = (kspace_shape[0] - target_shape[0]) // 2
  h_from = (kspace_shape[1] - target_shape[1]) // 2
  w_to = w_from + target_shape[0]
  h_to = h_from + target_shape[1]

  image = image[..., w_from:w_to, h_from:h_to, :]

  # complex_abs
  abs_image = tf.math.sqrt(tf.math.reduce_sum(image**2, axis=-1))

  # normalize_instance
  mean = tf.math.reduce_mean(abs_image)
  std = tf.math.reduce_std(abs_image)
  norm_image = (abs_image - mean) / std

  # clip_image
  image = tf.clip_by_value(norm_image, -6, 6)

  # process target
  norm_target = (target - mean) / std
  target = tf.clip_by_value(norm_target, -6, 6)

  return {
      'inputs': image,
      'targets': target,
      'mean': mean,
      'std': std,
      'volume_max': volume_max,
  }


def _h5_to_examples(path, log=False):
  """Yield MRI slices from an hdf5 file containing a single MRI volume."""
  if log:
    tf.print('fastmri_dataset._h5_to_examples call:',
             path,
             datetime.datetime.now().strftime('%H:%M:%S:%f'))
  with open(path, 'rb') as gf:
    with h5py.File(gf, 'r') as hf:
      # NOTE(dsuo): logic taken from reference code
      volume_max = hf.attrs.get('max', 0.0)

      for i in range(hf['kspace'].shape[0]):
        yield hf['kspace'][i], hf['kspace'][i].shape, hf['reconstruction_esc'][
            i], hf['reconstruction_esc'][i].shape, volume_max


def _create_generator(filename):
  signature = (
      tf.TensorSpec(shape=(640, None), dtype=tf.complex64),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(320, 320), dtype=tf.float32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(), dtype=tf.float32),
  )
  return tf.data.Dataset.from_generator(
      _h5_to_examples, args=(filename,), output_signature=signature)


def load_fastmri_split(global_batch_size,
                       split,
                       data_dir,
                       shuffle_rng,
                       num_batches,
                       repeat_final_eval_dataset):
  """Creates a split from the FastMRI dataset using tf.data.

  NOTE: only creates knee singlecoil datasets.
  NOTE: fastMRI has fixed randomness for eval.

  Args:
    global_batch_size: The global batch size returned by the data pipeline.
    split: One of ['train', 'eval_train', 'validation'].
    data_dir: The location of the data on disk.
    shuffle_rng: The RNG used to shuffle the split.
    num_batches: Number of batches to iterate over.
  Returns:
    A `tf.data.Dataset`.
  """
  if split not in ['train', 'eval_train', 'validation', 'test']:
    raise ValueError('Unrecognized split {}'.format(split))

  # Check if data directories exist because glob will not raise an error
  if not os.path.exists(os.path.join(data_dir, _TRAIN_DIR)):
    raise NotADirectoryError('Directory not found: {}'.format(
        os.path.join(data_dir, _TRAIN_DIR)))
  if not os.path.exists(os.path.join(data_dir, _VAL_DIR)):
    raise NotADirectoryError('Directory not found: {}'.format(
        os.path.join(data_dir, _VAL_DIR)))

  if split in ['train', 'eval_train']:
    file_pattern = os.path.join(data_dir, _TRAIN_DIR, '*.h5')
    h5_paths = glob.glob(file_pattern)
  elif split == 'validation':
    file_pattern = os.path.join(data_dir, _VAL_DIR, '*.h5')
    h5_paths = sorted(glob.glob(file_pattern))
  elif split == 'test':
    # The fastmri validation set is split into a validation and test set
    file_pattern = os.path.join(data_dir, _VAL_DIR, '*.h5')
    h5_paths = sorted(glob.glob(file_pattern))[100:]

  is_train = split == 'train'
  shuffle = is_train or split == 'eval_train'
  ds = tf.data.Dataset.from_tensor_slices(h5_paths)
  ds = ds.interleave(
      _create_generator,
      cycle_length=32,
      block_length=64,
      num_parallel_calls=16)
  if is_train:
    ds = ds.cache()

  def process_example(example_index, example):
    if shuffle:
      process_rng = tf.cast(jax.random.fold_in(shuffle_rng, 0), tf.int64)
      process_rng = tf.random.experimental.stateless_fold_in(
          process_rng, example_index)
    else:
      # NOTE(dsuo): we use fixed randomness for eval.
      process_rng = tf.cast(jax.random.PRNGKey(_EVAL_SEED), tf.int64)
    return _process_example(*example, process_rng)

  ds = ds.enumerate().map(process_example, num_parallel_calls=16)

  if shuffle:
    ds = ds.shuffle(
        16 * global_batch_size,
        seed=shuffle_rng[0],
        reshuffle_each_iteration=True)
    if is_train:
      ds = ds.repeat()

  ds = ds.batch(global_batch_size, drop_remainder=is_train)

  if is_train:
    ds = ds.prefetch(10)
    iterator = map(data_utils.shard_and_maybe_pad_np, ds)
    return iterator
  else:
    if num_batches:
      ds = ds.take(num_batches)
    ds = ds.cache()
    if repeat_final_eval_dataset:
      ds = ds.repeat()
    ds = ds.prefetch(10)
    return map(
        functools.partial(
            data_utils.shard_and_maybe_pad_np,
            global_batch_size=global_batch_size),
        ds)
