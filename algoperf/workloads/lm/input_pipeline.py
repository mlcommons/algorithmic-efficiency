"""Input pipeline for a LM dataset."""
import functools
import os

from datasets import Dataset, load_from_disk
from typing import Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from algoperf import data_utils
from algoperf.pytorch_utils import pytorch_setup

RANK = pytorch_setup()[1]
# Avoid multithreading in all processes but the first (rank 0).
AUTOTUNE = tf.data.AUTOTUNE if RANK == 0 else None


def get_lm_dataset(data_rng,
                   split: str,
                   data_dir: str,
                   is_training: bool,
                   vocab_size: int,
                   global_batch_size: int,
                   num_batches: Optional[int] = None,
                   repeat_final_dataset: bool = False,
                   vocab_path: Optional[str] = None):
  """Load HF dataset and return a TF dataset."""

  dataset_path = os.path.join(data_dir, split)
  dataset = load_from_disk(dataset_path)  # Loads HF arrow dataset

  is_training = split == "train"
  shuffle = split in ['train', 'eval_train']

  def tf_generator():
    """Generates data in a TensorFlow-friendly format."""
    for example in dataset:
      yield {
        "inputs": tf.convert_to_tensor(example["input_ids"][:-1], dtype=tf.int32),
        "targets": tf.convert_to_tensor(example["input_ids"][1:], dtype=tf.int32),
      }
  
  # Create a TensorFlow dataset from the generator function
  ds = tf.data.Dataset.from_generator(
        tf_generator,
        output_signature={
            "inputs": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "targets": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        }
    )

  # Avoid creating too many threads when using PyTorch DDP.
  if RANK != 0:
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    ds = ds.with_options(options)

  if shuffle:
    print(f"Shuffling dataset with seed: {data_rng[0]}, type={type(data_rng[0])}")
    ds = ds.shuffle(buffer_size=1024, seed=data_rng[0])

  if is_training:
    ds = ds.repeat()

  # Batch the dataset, ensuring the last batch is dropped if not full during training
  ds = ds.batch(global_batch_size, drop_remainder=is_training)
  ds = ds.prefetch(AUTOTUNE)

  # Limit the dataset to a fixed number of batches if `num_batches` is specified
  if num_batches:
    ds = ds.take(num_batches)

  # Shard the dataset across multiple GPUs/TPUs if necessary
  ds = map(
    functools.partial(
      data_utils.shard_and_maybe_pad_np,
      global_batch_size=global_batch_size),
    ds)

  return ds