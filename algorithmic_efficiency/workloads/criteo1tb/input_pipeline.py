"""Data loader for pre-processed Criteo data.

Similar to how the NVIDIA example works, we split data from the last day into a
validation and test split (taking the first half for test and second half for
validation). See here for the NVIDIA example:
https://github.com/NVIDIA/DeepLearningExamples/blob/4e764dcd78732ebfe105fc05ea3dc359a54f6d5e/PyTorch/Recommendation/DLRM/preproc/run_spark_cpu.sh#L119.
"""
import functools
import os
from typing import Optional

import tensorflow as tf

from algorithmic_efficiency import data_utils

_NUM_DAY_23_FILES = 36

# Raw vocab sizes from
# https://cloud.google.com/tpu/docs/tutorials/dlrm-dcn-2.x#run-model.
_VOCAB_SIZES = [
    39884406,
    39043,
    17289,
    7420,
    20263,
    3,
    7120,
    1543,
    63,
    38532951,
    2953546,
    403346,
    10,
    2208,
    11938,
    155,
    4,
    976,
    14,
    39979771,
    25641295,
    39664984,
    585935,
    12972,
    108,
    36
]


# Preprocessing is the same as
# https://github.com/mlcommons/inference/blob/master/recommendation/dlrm/tf/dataloader.py#L157
# and MAX_IND_RANGE used like
# https://github.com/facebookresearch/dlrm/blob/fbc37ebe21d4f88f18c6ae01333ada2d025e41cf/dlrm_data_pytorch.py#L298.
@tf.function
def _parse_example_fn(num_dense_features, example):
  """Parser function for pre-processed Criteo TSV records."""
  label_defaults = [[0.0]]
  int_defaults = [[0.0] for _ in range(num_dense_features)]
  categorical_defaults = [['00000000'] for _ in range(len(_VOCAB_SIZES))]
  record_defaults = label_defaults + int_defaults + categorical_defaults
  fields = tf.io.decode_csv(
      example, record_defaults, field_delim='\t', na_value='-1')

  num_labels = 1
  features = {}
  features['targets'] = tf.reshape(fields[0], (-1,))

  int_features = []
  for idx in range(num_dense_features):
    positive_val = tf.nn.relu(fields[idx + num_labels])
    int_features.append(tf.math.log(positive_val + 1))
  int_features = tf.stack(int_features, axis=1)

  cat_features = []
  for idx in range(len(_VOCAB_SIZES)):
    field = fields[idx + num_dense_features + num_labels]
    # We append the column index to the string to make the same id in different
    # columns unique.
    cat_features.append(
        tf.strings.to_hash_bucket_fast(field + str(idx), _VOCAB_SIZES[idx]))
  cat_features = tf.cast(
      tf.stack(cat_features, axis=1), dtype=int_features.dtype)
  features['inputs'] = tf.concat([int_features, cat_features], axis=1)
  return features


def get_criteo1tb_dataset(split: str,
                          shuffle_rng,
                          data_dir: str,
                          num_dense_features: int,
                          global_batch_size: int,
                          num_batches: Optional[int] = None,
                          repeat_final_dataset: bool = False):
  """Get the Criteo 1TB dataset for a given split."""
  num_test_files = _NUM_DAY_23_FILES // 2 + 1
  if split in ['train', 'eval_train']:
    file_paths = [os.path.join(data_dir, f'day_{d}_*') for d in range(0, 23)]
  elif split == 'validation':
    # Assumes files are of the format day_23_04.
    file_paths = [
        os.path.join(data_dir, f'day_23_{str(s).zfill(2)}')
        for s in range(num_test_files, _NUM_DAY_23_FILES)
    ]
  else:
    file_paths = [
        os.path.join(data_dir, f'day_23_{str(s).zfill(2)}')
        for s in range(0, num_test_files)
    ]

  is_training = split == 'train'
  shuffle = is_training or split == 'eval_train'
  ds = tf.data.Dataset.list_files(
      file_paths, shuffle=shuffle, seed=shuffle_rng[0])

  ds = ds.interleave(
      tf.data.TextLineDataset,
      cycle_length=128,
      block_length=global_batch_size // 8,
      num_parallel_calls=128,
      deterministic=False)
  if shuffle:
    ds = ds.shuffle(buffer_size=524_288 * 100, seed=shuffle_rng[1])
  ds = ds.batch(global_batch_size, drop_remainder=is_training)
  parse_fn = functools.partial(_parse_example_fn, num_dense_features)
  ds = ds.map(parse_fn, num_parallel_calls=16)
  if is_training:
    ds = ds.repeat()
  ds = ds.prefetch(10)

  if num_batches is not None:
    ds = ds.take(num_batches)

  # We do not use ds.cache() because the dataset is so large that it would OOM.
  if repeat_final_dataset:
    ds = ds.repeat()

  ds = map(
      functools.partial(
          data_utils.shard_and_maybe_pad_np,
          global_batch_size=global_batch_size),
      ds)

  return ds
