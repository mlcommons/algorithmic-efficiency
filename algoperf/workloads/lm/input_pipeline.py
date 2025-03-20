"""Input pipeline for a LM dataset."""
import functools
import os
from typing import Optional

from datasets import load_from_disk
import jax
import tensorflow as tf

from algoperf import data_utils
from algoperf.pytorch_utils import pytorch_setup

RANK = pytorch_setup()[1]
# Avoid multithreading in all processes but the first (rank 0).
# This ensures that only the primary process (RANK == 0) uses TensorFlow's
# automatic optimization (AUTOTUNE), while other processes disable it (None).
# tf.data.AUTOTUNE is a constant that lets TensorFlow automatically determine
# the optimal number of elements to prefetch or parallelize for dataset
# operations, improving performance.
AUTOTUNE = tf.data.AUTOTUNE if RANK == 0 else None


def get_lm_dataset(data_rng: jax.random.PRNGKey,
                   split: str,
                   data_dir: str,
                   vocab_size: int,
                   global_batch_size: int,
                   num_batches: Optional[int] = None,
                   repeat_final_dataset: bool = False,
                   vocab_path: Optional[str] = None):
  """Load HF dataset and return a TF dataset."""

  dataset_path = os.path.join(data_dir, split)
  dataset = load_from_disk(dataset_path)

  is_training = split == "train"
  shuffle = split in ['train', 'eval_train']

  dataset.set_format("tensorflow")  # tf.int64  # TODO (nico): is this needed?

  def tf_generator():
    """Generates data in a TensorFlow-friendly format."""
    for example in dataset:
      yield {
          "inputs": example["input_ids"][:-1],
          "targets": example["input_ids"][1:],
      }

  # Create a TensorFlow dataset
  ds = tf.data.Dataset.from_generator(
      tf_generator,
      output_signature={
          "inputs": tf.TensorSpec(shape=(None,), dtype=tf.int64),
          "targets": tf.TensorSpec(shape=(None,), dtype=tf.int64),
      })

  # Avoid creating too many threads when using PyTorch DDP.
  # Limits TensorFlow's threading for non-primary processes (RANK != 0)
  if RANK != 0:
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    ds = ds.with_options(options)

  if shuffle:
    ds = ds.shuffle(buffer_size=1024, seed=data_rng[0])

  if is_training:
    ds = ds.repeat()

  # Batch the dataset, grouping consecutive elements into fixed-size chunks.
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
