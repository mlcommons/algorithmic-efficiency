from flax import jax_utils
import jax
import tensorflow as tf

from algorithmic_efficiency import data_utils


def create_split(split,
                 dataset_builder,
                 rng,
                 global_batch_size,
                 train,
                 cache=False,
                 num_batches=None):
  """Creates a split from the ImageNet dataset using TensorFlow Datasets."""
  if split == 'eval_train':
    split = 'train'
  elif split == 'validation':
    split = 'test'

  shuffle_rng, preprocess_rng = jax.random.split(rng, 2)

  def preprocess_example(example):
      dtype = tf.float32
      # We call ds.enumerate() to get a globally unique per-example, per-step
      # index that we can fold into the RNG seed.
      (example_index, example) = example
      if train:
        per_step_preprocess_rng = tf.random.experimental.stateless_fold_in(
            tf.cast(preprocess_rng, tf.int64), example_index)
      
      return {'inputs': image, 'targets': example['targets']}

  ds = dataset_builder.as_dataset(split=split)
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if cache:
    ds = ds.cache()

  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * global_batch_size, seed=shuffle_rng[0])

  ds = ds.enumerate().map(
      lambda i,
      ex: preprocess_example((i, ex)),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(global_batch_size, drop_remainder=train)

  if num_batches is not None:
    ds = ds.take(num_batches)

  if repeat_final_dataset:
    ds = ds.repeat()

  ds = ds.prefetch(10)
  return ds


def create_input_iter(split,
                      dataset_builder,
                      rng,
                      global_batch_size,
                      train,
                      cache,
                      num_batches):
  ds = create_split(
      split,
      dataset_builder,
      rng,
      global_batch_size,
      train=train,
      cache=cache,
      num_batches=num_batches)
  it = map(data_utils.shard_numpy_ds, ds)

  # Note(Dan S): On a Nvidia 2080 Ti GPU, this increased GPU utilization by 10%.
  it = jax_utils.prefetch_to_device(it, 2)
  return it
