"""Data loader for pre-processed Criteo data."""
import os
from typing import Optional, Sequence

import jax
import tensorflow as tf
import csv
import numpy as np


def get_librispeech_dataset(split: str,
                          data_dir: str,
                          is_training: bool,
                          global_batch_size: int,
                          num_batches: Optional[int] = None,
                          repeat_final_dataset: bool = False):
  """Get the Librispeech  dataset for a given split."""
  feat_csv = '{}/{}.csv'.format(data_dir, split)
  print('feat_csv = ', feat_csv)
  print('path = ', os.getcwd())

  with open(feat_csv, newline='') as csvfile:
    data = list(csv.reader(csvfile))
  
  audio_list = []
  audio_paddings_list = []
  target_list = []
  target_paddings_list = []

  for example in data[1:]:
      audio = np.load('{}/{}/{}_audio.npy'.format(data_dir, split, example[1]))
      targets = np.load('{}/{}/{}_targets.npy'.format(data_dir, split, example[1]))

      audio_paddings = np.zeros_like(audio)
      audio_paddings = np.pad(audio_paddings, (0, 320000 - audio.shape[0]), constant_values=1.0)
      audio = np.pad(audio, (0, 320000 - audio.shape[0]), constant_values=0.0)
      
      target_paddings = np.zeros_like(targets)
      target_paddings = np.pad(target_paddings, (0, 256 - target_paddings.shape[0]), constant_values=1.0)
      targets = np.pad(targets, (0, 256 - targets.shape[0]), constant_values=0.0)

      audio_list.append(tf.constant(audio))
      audio_paddings_list.append(tf.constant(audio_paddings))

      target_list.append(tf.constant(targets))
      target_paddings_list.append(target_paddings)

  ds = tf.data.Dataset.from_tensor_slices({
      'inputs' : audio_list, 
      'input_paddings': audio_paddings_list, 
      'targets' : target_list, 
      'target_paddings' : target_paddings_list})

  if is_training:
    ds = ds.repeat()

  # TODO(sourabh2k15): we will need to select a validation split size that is evenly
  # divisible by the batch size.
  ds = ds.batch(global_batch_size, drop_remainder=False)
  if num_batches is not None:
    ds = ds.take(num_batches)
  if repeat_final_dataset:
    ds = ds.repeat()
  ds = ds.prefetch(tf.data.AUTOTUNE)
  
  return ds