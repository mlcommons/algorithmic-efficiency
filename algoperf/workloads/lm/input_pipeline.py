"""Input pipeline for a LM dataset."""
import functools
import os
from typing import Optional

import jax
import jax.numpy as jnp
import tensorflow as tf
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from algoperf import data_utils
from algoperf.pytorch_utils import pytorch_setup
from datasets import load_dataset
from datasets import load_from_disk

RANK = pytorch_setup()[1]
# Avoid multithreading in all processes but the first (rank 0).
# This ensures that only the primary process (RANK == 0) uses TensorFlow's
# automatic optimization (AUTOTUNE), while other processes disable it (None).
# tf.data.AUTOTUNE is a constant that lets TensorFlow automatically determine
# the optimal number of elements to prefetch or parallelize for dataset
# operations, improving performance.
AUTOTUNE = tf.data.AUTOTUNE if RANK == 0 else None


def get_hf_dataloader(cache_dir: str,
                      data_rng: jax.random.PRNGKey,
                      batch_size: int = 8,
                      seq_len: int = 32,
                      framework: str = "torch",
                      split="train"):
  """
    Create a data loader from HuggingFace's FineWeb dataset.

    Args:
        cache_dir: Directory to cache the dataset
        batch_size: Number of sequences per batch
        seq_len: Length of each sequence
        framework: Either "torch" or "jax" to specify output tensor type
        split: Dataset split to load
    """
  # Initialize tokenizer and get vocab size
  tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
  vocab_size = tokenizer.vocab_size
  # Load the FineWeb dataset in streaming mode
  fw = load_dataset(
      "HuggingFaceFW/fineweb-edu",
      name="sample-10BT",
      split=split,
      streaming=True,
      cache_dir=cache_dir)
  fw = fw.batch(batch_size=batch_size, drop_last_batch=True)
  if split in ['train', 'eval_train']:
    fw = fw.shuffle(seed=int(data_rng[-1]))

  def _tokenize(x):
    """Tokenize and pad text to seq_len+1 tokens."""
    if framework == "torch":
      tokens = tokenizer(x, return_tensors="pt")["input_ids"].squeeze()
      pad_length = seq_len - tokens.shape[0]
      if pad_length > 0:
        tokens = F.pad(tokens, pad_length, value=tokenizer.pad_token_id)
    elif framework == "jax":
      tokens = tokenizer(x, return_tensors="jax")["input_ids"].squeeze()
      pad_length = seq_len - tokens.shape[0]
      if pad_length > 0:
        tokens = jnp.pad(
            tokens,
            pad_length,
            mode="constant",
            constant_values=tokenizer.pad_token_id)
    return tokens[:seq_len + 1]

  def batch_iterator():
    for doc in fw:
      if framework == "torch":
        token_ids = torch.stack([_tokenize(x) for x in doc['text']])
        # Take first seq_len+1 tokens and convert to one-hot
        tokens = F.one_hot(token_ids, num_classes=vocab_size).float()
        # Split into input/target
        inputs, targets = tokens[:, :-1, :], tokens[:, 1:, :]
        inputs, targets = inputs.to("cuda"), targets.to("cuda")
      elif framework == "jax":
        token_ids = jnp.stack([_tokenize(x) for x in doc['text']])
        tokens = jax.nn.one_hot(token_ids, num_classes=vocab_size)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        inputs, targets = jax.device_put(inputs), jax.device_put(targets)
      yield {'inputs': inputs, 'targets': targets}

  return batch_iterator()


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
