"""Data loader for pre-processed librispeech data."""
import csv
from typing import Optional

from absl import logging
import numpy as np
import tensorflow as tf

from algorithmic_efficiency import spec


def get_librispeech_dataset(split_name: str,
                            data_dir: str,
                            shuffle_rng: spec.RandomState,
                            is_training: bool,
                            global_batch_size: int,
                            num_batches: Optional[int] = None):
  """Get the Librispeech  dataset for a given split."""
  splits = [split_name]

  if split_name.find('+') != -1:
    splits = split_name.split('+')

  ids = []

  for split in splits:
    logging.info(f'Loading split = {split}.')
    feat_csv = f'{data_dir}/{split}.csv'

    with open(feat_csv, newline='') as csvfile:
      data = list(csv.reader(csvfile))

    for example in data[1:]:
      ids.append(f'{split}/{example[1]}')

  def load_data(example_id):
    example_id = example_id.decode('utf-8')
    audio = np.load(f'{data_dir}/{example_id}_audio.npy')
    targets = np.load(f'{data_dir}/{example_id}_targets.npy')

    audio_paddings = np.zeros_like(audio, dtype=np.float32)
    audio_paddings = np.pad(
        audio_paddings, (0, 320000 - audio.shape[0]), constant_values=1.0)
    audio = np.pad(audio, (0, 320000 - audio.shape[0]), constant_values=0.0)

    target_paddings = np.zeros_like(targets, dtype=np.float32)
    target_paddings = np.pad(
        target_paddings, (0, 256 - target_paddings.shape[0]),
        constant_values=1.0)
    targets = np.pad(targets, (0, 256 - targets.shape[0]), constant_values=0)

    return audio, audio_paddings, targets, target_paddings

  def preprocess(example):
    example_id = example['ids']

    preprocessed_example = {}
    audio, audio_paddings, targets, target_paddings = tf.numpy_function(
        func=load_data,
        inp=[example_id],
        Tout=[tf.int64, tf.float32, tf.int32, tf.float32])

    # Make batches of tuples of (tensor, padding)
    preprocessed_example['inputs'] = (audio, audio_paddings)
    preprocessed_example['targets'] = (targets, target_paddings)

    return preprocessed_example

  ds = tf.data.Dataset.from_tensor_slices({'ids': ids})
  ds.shuffle(16 * global_batch_size, seed=shuffle_rng[0])

  ds = ds.map(preprocess, num_parallel_calls=10)

  if is_training:
    ds = ds.repeat()

  if split in ['train', 'eval_train']:
    ds = ds.shuffle(16 * global_batch_size, seed=shuffle_rng[0])

  ds = ds.batch(global_batch_size, drop_remainder=is_training)

  if is_training:
    ds = ds.repeat()

  if num_batches is not None:
    ds = ds.take(num_batches)

  ds = ds.prefetch(10)
  return ds
