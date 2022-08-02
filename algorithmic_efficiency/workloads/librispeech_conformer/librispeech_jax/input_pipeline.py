import jax
import numpy as np
import tensorflow_datasets as tfds


def _load_dataset(split, should_shuffle, data_rng, data_dir):
  """Loads a dataset split from TFDS."""
  if should_shuffle:
    file_data_rng, dataset_data_rng = jax.random.split(data_rng)
    file_data_rng = file_data_rng[0]
    dataset_data_rng = dataset_data_rng[0]
  else:
    file_data_rng = None
    dataset_data_rng = None

  read_config = tfds.ReadConfig(add_tfds_id=True, shuffle_seed=file_data_rng)
  dataset = tfds.load(
      'librispeech',
      split=split,
      shuffle_files=should_shuffle,
      read_config=read_config,
      data_dir=data_dir)

  if should_shuffle:
    dataset = dataset.shuffle(seed=dataset_data_rng, buffer_size=2**15)
    dataset = dataset.repeat()

  # We do not need to worry about repeating the dataset for evaluations because
  # we call itertools.cycle on the eval iterator, which stored the iterator in
  # memory to be repeated through.
  return dataset


def _get_batch_iterator(dataset_iter, global_batch_size, num_shards=None):
  """Turns a per-example iterator into a batched iterator.

  Constructs the batch from num_shards smaller batches, so that we can easily
  shard the batch to multiple devices during training. We use
  dynamic batching, so we specify some max number of graphs/nodes/edges, add
  as many graphs as we can, and then pad to the max values.

  Args:
    dataset_iter: The TFDS dataset iterator.
    global_batch_size: How many average-sized graphs go into the batch.
    num_shards: How many devices we should be able to shard the batch into.
  Yields:
    Batch in the init2winit format. Each field is a list of num_shards separate
    smaller batches.
  """



def get_dataset_iter(split, data_rng, data_dir, global_batch_size):
  ds = _load_dataset(
      split,
      should_shuffle=(split == 'train'),
      data_rng=data_rng,
      data_dir=data_dir)
  return _get_batch_iterator(iter(ds), global_batch_size)
