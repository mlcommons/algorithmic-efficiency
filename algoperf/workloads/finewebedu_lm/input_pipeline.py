"""Input pipeline for a LM dataset."""

import functools
import os
from typing import Optional

import jax
import tensorflow as tf
import datasets as hf_datasets

from algoperf import data_utils

AUTOTUNE = tf.data.experimental.AUTOTUNE
PAD_ID = tf.constant(-1, dtype=tf.int64)

TFDS_SPLIT_NAME = {'train': 'train', 'eval_train': 'train', 'validation': 'val'}

SEQUENCE_LENGTH = 1024
MAX_CORPUS_CHARS = 1_000_000_000
SHUFFLE_BUFFER_SIZE = 1000
VOCAB_SIZE = 50_257


def batch_with_padding(
  dataset: tf.data.Dataset,
  batch_size,
  padded_shapes=None,
  padding_id=PAD_ID,
):
  """Batches a tf.data.Dataset and adds padding if len(dataset) is not divisible by the batch size.

  Args:
    dataset: tf.data.Dataset
    batch_size: batch size of resulting batched dataset
    padded_shapes: shapes of the padded batches
    padding_id: value for padding, for elements in new batch

  Returns:
  """
  batched_dataset = dataset.batch(batch_size, drop_remainder=False)

  # tf.data.Dataset.padded.batch pads elements in the batch so we call it
  # again with batch_size=1 to pad each element in original batch.
  padded_batched_dataset = batched_dataset.padded_batch(
    1, padded_shapes=padded_shapes, padding_values=padding_id
  )

  # Remove extra dimension resulting from the batch_size=1.
  padded_batched_dataset = padded_batched_dataset.unbatch()

  return padded_batched_dataset


def get_data_iter(
  data_rng: jax.random.PRNGKey,
  split: str,
  data_dir: str,
  batch_size: int,
  num_batches: Optional[int] = None,
):
  ds = get_lm_dataset(data_rng, split, data_dir, batch_size, num_batches)

  it = map(
    functools.partial(
      data_utils.shard_and_maybe_pad_np, global_batch_size=batch_size
    ),
    ds,
  )

  return iter(it)


def get_lm_dataset(
  data_rng: jax.random.PRNGKey,
  split: str,
  data_dir: str,
  batch_size: int,
  num_batches: Optional[int] = None,
):
  """Load preprocessed TF dataset."""
  if split not in TFDS_SPLIT_NAME:
    raise NotImplementedError

  shuffle_seed = jax.random.randint(data_rng, (), -(2**31), 2**31 - 1)

  data_dir = os.path.join(data_dir, TFDS_SPLIT_NAME[split])
  ds = hf_datasets.load_from_disk(data_dir)
  tokens_ds = ds.to_tf_dataset()

  # tokens
  tokens_ds = tokens_ds.flat_map(tf.data.Dataset.from_tensor_slices)

  # sequences
  sequences_ds = tokens_ds.batch(SEQUENCE_LENGTH + 1, drop_remainder=True)

  # get inputs and outputs
  sequences_ds = sequences_ds.map(
    lambda x: {
      'inputs': x['input_ids'][:SEQUENCE_LENGTH],
      'targets': x['input_ids'][1:],
    },
    num_parallel_calls=AUTOTUNE,
  )
  if split == 'train':
    ds = sequences_ds.shuffle(SHUFFLE_BUFFER_SIZE, seed=shuffle_seed)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.take(num_batches) if num_batches is not None else ds
    ds = ds.repeat()
    ds = ds.map(
      lambda x: {
        'inputs': x['inputs'],
        'targets': x['targets'],
        'weights': None,
      }
    )
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  elif split == 'eval_train':
    ds = batch_with_padding(
      sequences_ds,
      batch_size,
      padded_shapes={
        'inputs': (batch_size, None),
        'targets': (batch_size, None),
      },
    )
    ds = ds.take(num_batches) if num_batches is not None else ds
    ds = ds.repeat()
    ds = ds.map(
      lambda x: {
        'inputs': x['inputs'],
        'targets': x['targets'],
        'weights': tf.where(tf.equal(x['inputs'], PAD_ID), 0.0, 1.0),
      }
    )
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  elif split == 'validation':
    ds = batch_with_padding(
      sequences_ds,
      batch_size,
      padded_shapes={
        'inputs': (batch_size, None),
        'targets': (batch_size, None),
      },
    )
    ds = ds.take(num_batches) if num_batches is not None else ds
    ds = ds.repeat()
    ds = ds.map(
      lambda x: {
        'inputs': x['inputs'],
        'targets': x['targets'],
        'weights': tf.where(tf.equal(x['inputs'], PAD_ID), 0.0, 1.0),
      }
    )
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds
