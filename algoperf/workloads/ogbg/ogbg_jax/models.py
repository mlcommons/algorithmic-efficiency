# Forked from the init2winit implementation here
# https://github.com/google/init2winit/blob/master/init2winit/model_lib/gnn.py.
from typing import Tuple

from flax import linen as nn
import jax.numpy as jnp
import jraph

from algoperf.jax_utils import Dropout

DROPOUT_RATE = 0.1


def _make_embed(latent_dim, name):

  def make_fn(inputs):
    return nn.Dense(features=latent_dim, name=name)(inputs)

  return make_fn


def _make_mlp(hidden_dims, activation_fn, train, dropout_rate=DROPOUT_RATE):
  """Creates a MLP with specified dimensions."""

  @jraph.concatenated_args
  def make_fn(inputs):
    x = inputs
    for dim in hidden_dims:
      x = nn.Dense(features=dim)(x)
      x = nn.LayerNorm()(x)
      x = activation_fn(x)
      x = Dropout(
          rate=dropout_rate, deterministic=not train)(
              x, rate=dropout_rate)
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
  num_message_passing_steps: int = 5
  activation_fn_name: str = 'relu'

  @nn.compact
  def __call__(self, graph, train, dropout_rate=DROPOUT_RATE):

    graph = graph._replace(
        globals=jnp.zeros([graph.n_node.shape[0], self.num_outputs]))

    embedder = jraph.GraphMapFeatures(
        embed_node_fn=_make_embed(self.latent_dim, name='node_embedding'),
        embed_edge_fn=_make_embed(self.latent_dim, name='edge_embedding'))
    graph = embedder(graph)

    if self.activation_fn_name == 'relu':
      activation_fn = nn.relu
    elif self.activation_fn_name == 'gelu':
      activation_fn = nn.gelu
    elif self.activation_fn_name == 'silu':
      activation_fn = nn.silu
    else:
      raise ValueError(
          f'Invalid activation function name: {self.activation_fn_name}')

    for _ in range(self.num_message_passing_steps):
      net = jraph.GraphNetwork(
          update_edge_fn=_make_mlp(
              self.hidden_dims,
              activation_fn=activation_fn,
              train=train,
              dropout_rate=dropout_rate),
          update_node_fn=_make_mlp(
              self.hidden_dims,
              activation_fn=activation_fn,
              train=train,
              dropout_rate=dropout_rate),
          update_global_fn=_make_mlp(
              self.hidden_dims,
              activation_fn=activation_fn,
              train=train,
              dropout_rate=dropout_rate))

      graph = net(graph)

    # Map globals to represent the final result
    decoder = jraph.GraphMapFeatures(embed_global_fn=nn.Dense(self.num_outputs))
    graph = decoder(graph)

    return graph.globals
