# Forked from the init2winit implementation here
# https://github.com/google/init2winit/blob/master/init2winit/model_lib/gnn.py.
from typing import Optional, Tuple

from flax import linen as nn
import jax.numpy as jnp
import jraph


def _make_embed(latent_dim, name):

  def make_fn(inputs):
    return nn.Dense(features=latent_dim, name=name)(inputs)

  return make_fn


def _make_mlp(hidden_dims, dropout):
  """Creates a MLP with specified dimensions."""

  @jraph.concatenated_args
  def make_fn(inputs):
    x = inputs
    for dim in hidden_dims:
      x = nn.Dense(features=dim)(x)
      x = nn.LayerNorm()(x)
      x = nn.relu(x)
      x = dropout(x)
    return x

  return make_fn


class GNN(nn.Module):
  """Defines a graph network.
  The model assumes the input data is a jraph.GraphsTuple without global
  variables. The final prediction will be encoded in the globals.
  """
  num_outputs: int
  latent_dim: int = 256
  hidden_dims: Tuple[int] = (256,)
  # If None, defaults to 0.1.
  dropout_rate: Optional[float] = 0.1
  num_message_passing_steps: int = 5

  @nn.compact
  def __call__(self, graph, train):
    if self.dropout_rate is None:
      dropout_rate = 0.1
    else:
      dropout_rate = self.dropout_rate
    dropout = nn.Dropout(rate=dropout_rate, deterministic=not train)

    graph = graph._replace(
        globals=jnp.zeros([graph.n_node.shape[0], self.num_outputs]))

    embedder = jraph.GraphMapFeatures(
        embed_node_fn=_make_embed(self.latent_dim, name='node_embedding'),
        embed_edge_fn=_make_embed(self.latent_dim, name='edge_embedding'))
    graph = embedder(graph)

    for _ in range(self.num_message_passing_steps):
      net = jraph.GraphNetwork(
          update_edge_fn=_make_mlp(self.hidden_dims, dropout=dropout),
          update_node_fn=_make_mlp(self.hidden_dims, dropout=dropout),
          update_global_fn=_make_mlp(self.hidden_dims, dropout=dropout))

      graph = net(graph)

    # Map globals to represent the final result
    decoder = jraph.GraphMapFeatures(embed_global_fn=nn.Dense(self.num_outputs))
    graph = decoder(graph)

    return graph.globals
