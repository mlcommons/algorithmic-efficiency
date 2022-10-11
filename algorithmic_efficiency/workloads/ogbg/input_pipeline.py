# Forked from Flax example which can be found here:
# https://github.com/google/flax/blob/main/examples/ogbg_molpcba/input_pipeline.py
# and from the init2winit fork here
# https://github.com/google/init2winit/blob/master/init2winit/dataset_lib/ogbg_molpcba.py
"""Exposes the ogbg-molpcba dataset in a convenient format."""

import jax
import jraph
import numpy as np
import tensorflow_datasets as tfds

AVG_NODES_PER_GRAPH = 26
AVG_EDGES_PER_GRAPH = 56


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
      'ogbg_molpcba:0.1.2',
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


def _to_jraph(example):
  """Converts an example graph to jraph.GraphsTuple."""
  example = jax.tree_map(lambda x: x._numpy(), example)  # pylint: disable=protected-access
  edge_feat = example['edge_feat']
  node_feat = example['node_feat']
  edge_index = example['edge_index']
  labels = example['labels']
  num_nodes = example['num_nodes']

  senders = edge_index[:, 0]
  receivers = edge_index[:, 1]

  return jraph.GraphsTuple(
      n_node=num_nodes,
      n_edge=np.array([len(edge_index) * 2]),
      nodes=node_feat,
      edges=np.concatenate([edge_feat, edge_feat]),
      # Make the edges bidirectional
      senders=np.concatenate([senders, receivers]),
      receivers=np.concatenate([receivers, senders]),
      # Keep the labels with the graph for batching. They will be removed
      # in the processed batch.
      globals=np.expand_dims(labels, axis=0))


def _get_weights_by_nan_and_padding(labels, padding_mask):
  """Handles NaNs and padding in labels.

  Sets all the weights from examples coming from padding to 0. Changes all NaNs
  in labels to 0s and sets the corresponding per-label weight to 0.

  Args:
    labels: Labels including labels from padded examples
    padding_mask: Binary array of which examples are padding
  Returns:
    tuple of (processed labels, corresponding weights)
  """
  nan_mask = np.isnan(labels)
  replaced_labels = np.copy(labels)
  np.place(replaced_labels, nan_mask, 0)

  weights = 1.0 - nan_mask
  # Weights for all labels of a padded element will be 0
  weights = weights * padding_mask[:, None]
  return replaced_labels, weights


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
  if not num_shards:
    num_shards = jax.local_device_count()

  # We will construct num_shards smaller batches and then put them together.
  per_device_batch_size = global_batch_size // num_shards

  max_n_nodes = AVG_NODES_PER_GRAPH * per_device_batch_size
  max_n_edges = AVG_EDGES_PER_GRAPH * per_device_batch_size
  max_n_graphs = per_device_batch_size

  jraph_iter = map(_to_jraph, dataset_iter)
  batched_iter = jraph.dynamically_batch(jraph_iter,
                                         max_n_nodes + 1,
                                         max_n_edges,
                                         max_n_graphs + 1)

  count = 0
  graphs_shards = []
  labels_shards = []
  weights_shards = []

  for batched_graph in batched_iter:
    count += 1

    # Separate the labels from the graph
    labels = batched_graph.globals
    graph = batched_graph._replace(globals={})

    replaced_labels, weights = _get_weights_by_nan_and_padding(
        labels, jraph.get_graph_padding_mask(graph))

    graphs_shards.append(graph)
    labels_shards.append(replaced_labels)
    weights_shards.append(weights)

    if count == num_shards:

      def f(x):
        return jax.tree_map(lambda *vals: np.stack(vals, axis=0), x[0], *x[1:])

      graphs_shards = f(graphs_shards)
      labels_shards = f(labels_shards)
      weights_shards = f(weights_shards)
      yield {
          'inputs': graphs_shards,
          'targets': labels_shards,
          'weights': weights_shards
      }

      count = 0
      graphs_shards = []
      labels_shards = []
      weights_shards = []


def get_dataset_iter(split, data_rng, data_dir, global_batch_size):
  ds = _load_dataset(
      split,
      should_shuffle=(split == 'train'),
      data_rng=data_rng,
      data_dir=data_dir)
  return _get_batch_iterator(iter(ds), global_batch_size)
