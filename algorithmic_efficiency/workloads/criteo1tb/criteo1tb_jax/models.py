"""A JAX implementation of DLRM-Small."""

from typing import Sequence

import flax.linen as nn
from jax import nn as jnn
import jax.numpy as jnp


def dot_interact(concat_features):
  """Performs feature interaction operation between dense or sparse features.
  Input tensors represent dense or sparse features.
  Pre-condition: The tensors have been stacked along dimension 1.
  Args:
    concat_features: Array of features with shape [B, n_features, feature_dim].
  Returns:
    activations: Array representing interacted features.
  """
  batch_size = concat_features.shape[0]

  # Interact features, select upper or lower-triangular portion, and re-shape.
  xactions = jnp.matmul(concat_features,
                        jnp.transpose(concat_features, [0, 2, 1]))
  feature_dim = xactions.shape[-1]

  indices = jnp.array(jnp.triu_indices(feature_dim))
  num_elems = indices.shape[1]
  indices = jnp.tile(indices, [1, batch_size])
  indices0 = jnp.reshape(
      jnp.tile(jnp.reshape(jnp.arange(batch_size), [-1, 1]), [1, num_elems]),
      [1, -1])
  indices = tuple(jnp.concatenate((indices0, indices), 0))
  activations = xactions[indices]
  activations = jnp.reshape(activations, [batch_size, -1])
  return activations


class DlrmSmall(nn.Module):
  """Define a DLRM-Small model.

  Parameters:
    vocab_size: vocab size of embedding table.
    num_dense_features: number of dense features as the bottom mlp input.
    mlp_bottom_dims: dimensions of dense layers of the bottom mlp.
    mlp_top_dims: dimensions of dense layers of the top mlp.
    embed_dim: embedding dimension.
  """

  vocab_size: int = 32 * 128 * 1024  # 4_194_304
  num_dense_features: int = 13
  mlp_bottom_dims: Sequence[int] = (512, 256, 128)
  mlp_top_dims: Sequence[int] = (1024, 1024, 512, 256, 1)
  embed_dim: int = 128
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x, train):
    bot_mlp_input, cat_features = jnp.split(x, [self.num_dense_features], 1)
    cat_features = jnp.asarray(cat_features, dtype=jnp.int32)

    # Bottom MLP.
    for dense_dim in self.mlp_bottom_dims:
      bot_mlp_input = nn.Dense(
          dense_dim,
          kernel_init=jnn.initializers.glorot_uniform(),
          bias_init=jnn.initializers.normal(stddev=jnp.sqrt(1.0 / dense_dim)),
      )(
          bot_mlp_input)
      bot_mlp_input = nn.relu(bot_mlp_input)
    bot_mlp_output = bot_mlp_input
    batch_size = bot_mlp_output.shape[0]
    feature_stack = jnp.reshape(bot_mlp_output,
                                [batch_size, -1, self.embed_dim])

    # Embedding table look-up.
    idx_lookup = jnp.reshape(cat_features, [-1]) % self.vocab_size

    def scaled_init(key, shape, dtype=jnp.float_):
      return (jnn.initializers.uniform(scale=1.0)(key, shape, dtype) /
              jnp.sqrt(self.vocab_size))

    embedding_table = self.param('embedding_table',
                                 scaled_init, [self.vocab_size, self.embed_dim])

    idx_lookup = jnp.reshape(idx_lookup, [-1])
    embed_features = embedding_table[idx_lookup]
    embed_features = jnp.reshape(embed_features,
                                 [batch_size, -1, self.embed_dim])
    feature_stack = jnp.concatenate([feature_stack, embed_features], axis=1)
    dot_interact_output = dot_interact(concat_features=feature_stack)
    top_mlp_input = jnp.concatenate([bot_mlp_output, dot_interact_output],
                                    axis=-1)
    mlp_input_dim = top_mlp_input.shape[1]
    mlp_top_dims = self.mlp_top_dims
    num_layers_top = len(mlp_top_dims)
    for layer_idx, fan_out in enumerate(mlp_top_dims):
      fan_in = mlp_input_dim if layer_idx == 0 else mlp_top_dims[layer_idx - 1]
      top_mlp_input = nn.Dense(
          fan_out,
          kernel_init=jnn.initializers.normal(
              stddev=jnp.sqrt(2.0 / (fan_in + fan_out))),
          bias_init=jnn.initializers.normal(stddev=jnp.sqrt(1.0 / fan_out)))(
              top_mlp_input)
      if layer_idx < (num_layers_top - 1):
        top_mlp_input = nn.relu(top_mlp_input)
      if self.dropout_rate > 0.0 and layer_idx == num_layers_top - 2:
        top_mlp_input = nn.Dropout(
            rate=self.dropout_rate, deterministic=not train)(
                top_mlp_input)
    logits = top_mlp_input
    return logits
