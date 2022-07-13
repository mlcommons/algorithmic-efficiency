"""Data loader for pre-processed Criteo data."""
import os
from typing import Optional, Sequence

import jax
import tensorflow as tf


def get_criteo1tb_dataset(split: str,
                          data_dir: str,
                          is_training: bool,
                          global_batch_size: int,
                          num_dense_features: int,
                          vocab_sizes: Sequence[int],
                          num_batches: Optional[int] = None,
                          repeat_final_dataset: bool = False):
  """Get the Criteo 1TB dataset for a given split."""
  if split in ['train', 'eval_train']:
    file_path = os.path.join(data_dir, 'train/train*')
  else:
    file_path = os.path.join(data_dir, 'eval/eval*')
  num_devices = jax.local_device_count()
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

    num_labels = 1
    num_dense = len(int_defaults)
    features = {}
    features['targets'] = tf.reshape(fields[0], [per_device_batch_size])

    int_features = []
    for idx in range(num_dense):
      int_features.append(tf.math.log(fields[idx + num_labels] + 1))
    int_features = tf.stack(int_features, axis=1)

    cat_features = []
    for idx in range(len(vocab_sizes)):
      cat_features.append(
              tf.io.decode_raw(fields[idx + num_dense + num_labels], tf.int64)[:, 0] % vocab_sizes[idx])
    print(cat_features)
    cat_features = tf.cast(
        tf.stack(cat_features, axis=1), dtype=int_features.dtype)
    features['inputs'] = tf.concat([int_features, cat_features], axis=1)
    features['weights'] = tf.ones(
        shape=(features['inputs'].shape[0],), dtype=features['inputs'].dtype)
    return features

  ds = tf.data.Dataset.list_files(file_path, shuffle=False)
  if is_training:
    ds = ds.repeat()
  ds = ds.interleave(
      tf.data.TextLineDataset,
      cycle_length=128,
      block_length=per_device_batch_size // 8,
      num_parallel_calls=128,
      deterministic=False)
  # TODO(znado): we will need to select a validation split size that is evenly
  # divisible by the batch size.
  ds = ds.batch(per_device_batch_size, drop_remainder=True)
  ds = ds.map(_parse_example_fn, num_parallel_calls=16)
  if num_batches is not None:
    ds = ds.take(num_batches)
  if repeat_final_dataset:
    ds = ds.repeat()
  ds = ds.prefetch(tf.data.AUTOTUNE)
  ds = ds.batch(num_devices)
  return ds
