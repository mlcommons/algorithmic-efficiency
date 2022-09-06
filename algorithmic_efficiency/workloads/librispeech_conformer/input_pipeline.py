"""Data loader for pre-processed Criteo data."""
import os
from typing import Optional, Sequence

import jax
from flax import jax_utils
import tensorflow as tf
import csv
import numpy as np
from absl import logging
from algorithmic_efficiency import spec
from algorithmic_efficiency import data_utils

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
    logging.info('loading split = {}'.format(split))
    feat_csv = '{}/{}.csv'.format(data_dir, split)

    with open(feat_csv, newline='') as csvfile:
      data = list(csv.reader(csvfile))

    for example in data[1:]:
      ids.append('{}/{}'.format(split, example[1]))

  def load_data(id):
    id = id.decode('utf-8') 
    audio = np.load('{}/{}_audio.npy'.format(data_dir, id))
    targets = np.load('{}/{}_targets.npy'.format(data_dir, id))

    audio_paddings = np.zeros_like(audio, dtype=np.float32)
    audio_paddings = np.pad(audio_paddings, (0, 320000 - audio.shape[0]), constant_values=1.0)
    audio = np.pad(audio, (0, 320000 - audio.shape[0]), constant_values=0.0)
      
    target_paddings = np.zeros_like(targets, dtype=np.float32)
    target_paddings = np.pad(target_paddings, (0, 256 - target_paddings.shape[0]), constant_values=1.0)
    targets = np.pad(targets, (0, 256 - targets.shape[0]), constant_values=0)

    return audio, audio_paddings, targets, target_paddings

  def preprocess(example):
    id = example['ids']

    preprocessed_example = {}
    audio, audio_paddings, targets, target_paddings = tf.numpy_function(
        func=load_data,
        inp=[id],
        Tout=[tf.int64, tf.float32, tf.int32, tf.float32])
    
    preprocessed_example['inputs'] = audio
    preprocessed_example['input_paddings'] = audio_paddings
    preprocessed_example['targets'] = targets
    preprocessed_example['target_paddings'] = target_paddings

    return preprocessed_example

  ds = tf.data.Dataset.from_tensor_slices({
      'ids' : ids
  })

  options = tf.data.Options()
  options.threading.private_threadpool_size = 48

  ds = ds.with_options(options)  
  ds = ds.map(preprocess)

  if is_training:
    ds = ds.repeat()
    ds = ds.shuffle(16 * global_batch_size, seed=shuffle_rng[0])

  # TODO(sourabh2k15): we will need to select a validation split size that is evenly
  # divisible by the batch size.
  ds = ds.batch(global_batch_size, drop_remainder=is_training)

  if is_training:
    ds = ds.repeat()
  
  if num_batches is not None:
    ds = ds.take(num_batches)

  ds = ds.prefetch(10)
  return ds
