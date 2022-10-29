"""Data loader for pre-processed Criteo data.

Similar to how the NVIDIA example works, we split data from the last day into a
validation and test split (taking the first half for test and second half for
validation). See here for the NVIDIA example:
https://github.com/NVIDIA/DeepLearningExamples/blob/4e764dcd78732ebfe105fc05ea3dc359a54f6d5e/PyTorch/Recommendation/DLRM/preproc/run_spark_cpu.sh#L119.
"""
import functools
import math
import os
from typing import Optional, Sequence

import jax
import tensorflow as tf
import torch

from algorithmic_efficiency import data_utils

_CSV_LINES_PER_FILE = 5_000_000


def get_criteo1tb_dataset(split: str,
                          data_dir: str,
                          global_batch_size: int,
                          num_dense_features: int,
                          vocab_sizes: Sequence[int],
                          num_batches: Optional[int] = None,
                          repeat_final_dataset: bool = False):
  """Get the Criteo 1TB dataset for a given split."""
  if split in ['train', 'eval_train']:
    file_path = os.path.join(data_dir, 'day_[0-22]_*')
  else:
    file_path = os.path.join(data_dir, 'day_23_*')
  num_devices = max(torch.cuda.device_count(), jax.local_device_count())
  per_device_batch_size = global_batch_size // num_devices

  @tf.function
  def _parse_example_fn(example):
    """Parser function for pre-processed Criteo TSV records."""
    label_defaults = [[0.0]]
    int_defaults = [[0.0] for _ in range(num_dense_features)]
    categorical_defaults = [['00000000'] for _ in range(len(vocab_sizes))]
    record_defaults = label_defaults + int_defaults + categorical_defaults
    fields = tf.io.decode_csv(
        example, record_defaults, field_delim='\t', na_value='-1')

    targets = tf.expand_dims(fields[0], axis=1)  # (batch, 1)
    features = {'targets': targets}

    num_labels = 1
    num_dense = len(int_defaults)
    int_features = []
    for idx in range(num_dense):
      positive_val = tf.nn.relu(fields[idx + num_labels])
      int_features.append(tf.math.log(positive_val + 1))
    int_features = tf.stack(int_features, axis=1)

    cat_features = []
    for idx in range(len(vocab_sizes)):
      cat_features.append(
          tf.io.decode_raw(fields[idx + num_dense + num_labels], tf.int64)[:, 0]
          % vocab_sizes[idx])
    cat_features = tf.cast(
        tf.stack(cat_features, axis=1), dtype=int_features.dtype)
    features['inputs'] = tf.concat([int_features, cat_features], axis=1)
    return features

  ds = tf.data.Dataset.list_files(file_path, shuffle=False)
  # There should be 36 files for day_23.csv, sp we take the first half of them
  # as the test split and the second half as the validation split.
  if split == 'test':
    ds = ds.take(18)
  elif split == 'validation':
    ds = ds.skip(18)

  is_training = split == 'train'
  # For evals, we only need a few files worth of data, so we only load those
  # that we need.
  if not is_training and num_batches is not None:
    num_examples = num_batches * global_batch_size
    num_files = math.ceil(num_examples / _CSV_LINES_PER_FILE)
    ds = ds.take(num_files)

  if is_training:
    ds = ds.repeat()
  ds = ds.interleave(
      tf.data.TextLineDataset,
      cycle_length=32,
      block_length=per_device_batch_size,
      num_parallel_calls=32,
      deterministic=False)
  ds = ds.batch(global_batch_size, drop_remainder=is_training)
  ds = ds.map(_parse_example_fn, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.prefetch(tf.data.AUTOTUNE)

  if num_batches is not None:
    ds = ds.take(num_batches)

  # We do not need a ds.cache() because we will do this anyways with
  # itertools.cycle in the base workload.
  if repeat_final_dataset:
    ds = ds.repeat()

  ds = map(
      functools.partial(
          data_utils.shard_and_maybe_pad_np, global_batch_size=global_batch_size),
      ds)

  return ds
