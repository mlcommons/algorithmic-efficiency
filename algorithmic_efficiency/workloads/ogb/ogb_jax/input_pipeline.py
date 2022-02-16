# Forked from Flax example which can be found here:
# https://github.com/google/flax/blob/main/examples/ogbg_molpcba/input_pipeline.py

"""Exposes the ogbg-molpcba dataset in a convenient format."""

import functools
from typing import Any, Dict, NamedTuple
import jax
import jraph
import numpy as np
import tensorflow as tf
# Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.experimental.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds


AVG_NODES_PER_GRAPH = 26
AVG_EDGES_PER_GRAPH = 56


class GraphsTupleSize(NamedTuple):
  """Helper class to represent padding and graph sizes."""
  n_node: int
  n_edge: int
  n_graph: int


def _get_raw_dataset(
  split_name: str,
  data_dir: str,
  file_shuffle_seed: Any) -> Dict[str, tf.data.Dataset]:
  """Returns datasets as tf.data.Dataset, organized by split."""
  ds_builder = tfds.builder('ogbg_molpcba', data_dir=data_dir)
  ds_builder.download_and_prepare()
  config = tfds.ReadConfig(shuffle_seed=file_shuffle_seed)
  return ds_builder.as_dataset(split=split_name, read_config=config)


def convert_to_graphs_tuple(graph: Dict[str, tf.Tensor],
                            add_virtual_node: bool,
                            add_undirected_edges: bool,
                            add_self_loops: bool) -> jraph.GraphsTuple:
  """Converts a dictionary of tf.Tensors to a GraphsTuple."""
  num_nodes = tf.squeeze(graph['num_nodes'])
  num_edges = tf.squeeze(graph['num_edges'])
  nodes = graph['node_feat']
  edges = graph['edge_feat']
  labels = graph['labels']
  senders = graph['edge_index'][:, 0]
  receivers = graph['edge_index'][:, 1]

  return jraph.GraphsTuple(
      n_node=tf.expand_dims(num_nodes, 0),
      n_edge=tf.expand_dims(num_edges, 0),
      nodes=nodes,
      edges=edges,
      senders=senders,
      receivers=receivers,
      globals=tf.expand_dims(labels, axis=0))


def _get_valid_mask(graphs: jraph.GraphsTuple):
  """Gets the binary mask indicating only valid labels and graphs."""
  labels = graphs.globals
  # We have to ignore all NaN values - which indicate labels for which
  # the current graphs have no label.
  labels_masks = ~np.isnan(labels)

  # Since we have extra 'dummy' graphs in our batch due to padding, we want
  # to mask out any loss associated with the dummy graphs.
  # Since we padded with `pad_with_graphs` we can recover the mask by using
  # get_graph_padding_mask.
  graph_masks = jraph.get_graph_padding_mask(graphs)

  # Combine the mask over labels with the mask over graphs.
  masks = labels_masks & graph_masks[:, None]
  graphs = graphs._replace(globals=[])
  return graphs, labels, masks


def _batch_for_pmap(iterator):
  graphs = []
  labels = []
  masks = []
  count = 0
  for batch in iterator:
    graph_batch, label_batch, mask_batch = _get_valid_mask(batch)
    count += 1
    graphs.append(graph_batch)
    labels.append(label_batch)
    masks.append(mask_batch)
    if count == jax.local_device_count():
      graphs = jax.tree_multimap(lambda *x: np.stack(x, axis=0), *graphs)
      labels = np.stack(labels)
      masks = np.stack(masks)
      yield graphs, labels, masks
      graphs = []
      labels = []
      masks = []
      count = 0


def get_dataset_iter(split_name: str,
                     data_rng: jax.random.PRNGKey,
                     data_dir: str,
                     batch_size: int,
                     add_virtual_node: bool = True,
                     add_undirected_edges: bool = True,
                     add_self_loops: bool = True) -> Dict[str, tf.data.Dataset]:
  """Returns datasets of batched GraphsTuples, organized by split."""
  if batch_size <= 1:
    raise ValueError('Batch size must be > 1 to account for padding graphs.')

  file_shuffle_seed, dataset_shuffle_seed = jax.random.split(data_rng)
  file_shuffle_seed = file_shuffle_seed[0]
  dataset_shuffle_seed = dataset_shuffle_seed[0]

  # Obtain the original datasets.
  dataset = _get_raw_dataset(split_name, data_dir, file_shuffle_seed)

  # Construct the GraphsTuple converter function.
  convert_to_graphs_tuple_fn = functools.partial(
      convert_to_graphs_tuple,
      add_virtual_node=add_self_loops,
      add_undirected_edges=add_undirected_edges,
      add_self_loops=add_virtual_node,
  )

  dataset = dataset.map(
      convert_to_graphs_tuple_fn,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=True)

  # Repeat and shuffle the training split.
  if split_name == 'train':
    dataset = dataset.shuffle(
        buffer_size=2**15,
        seed=dataset_shuffle_seed,
        reshuffle_each_iteration=True)
    dataset = dataset.repeat()
  # We do not need to cache the validation and test sets because we do this
  # later with itertools.cycle.

  # Batch and pad each split. Note that this also converts the graphs to numpy.
  max_n_nodes = AVG_NODES_PER_GRAPH * batch_size
  max_n_edges = AVG_EDGES_PER_GRAPH * batch_size
  batched_iter = jraph.dynamically_batch(
      graphs_tuple_iterator=iter(dataset),
      n_node=max_n_nodes,
      n_edge=max_n_edges,
      n_graph=batch_size)

  # An iterator the same as above, but where each element has an extra leading
  # dim of size jax.local_device_count().
  pmapped_iterator = _batch_for_pmap(batched_iter)
  return pmapped_iterator
