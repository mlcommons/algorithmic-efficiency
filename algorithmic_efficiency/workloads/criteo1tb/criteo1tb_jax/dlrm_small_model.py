"""A JAX implementation of DLRM-Small."""

import functools
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
    xactions = jnp.matmul(concat_features, jnp.transpose(concat_features, [0, 2, 1]))
    feature_dim = xactions.shape[-1]

    indices = jnp.array(jnp.triu_indices(feature_dim))
    num_elems = indices.shape[1]
    indices = jnp.tile(indices, [1, batch_size])
    indices0 = jnp.reshape(
        jnp.tile(jnp.reshape(jnp.arange(batch_size), [-1, 1]), [1, num_elems]), [1, -1]
    )
    indices = tuple(jnp.concatenate((indices0, indices), 0))
    activations = xactions[indices]
    activations = jnp.reshape(activations, [batch_size, -1])
    return activations


class DlrmSmall(nn.Module):
    """Define a DLRM-Small model.

    Parameters:
      vocab_sizes: list of vocab sizes of embedding tables.
      total_vocab_sizes: sum of embedding table sizes (for jit compilation).
      mlp_bottom_dims: dimensions of dense layers of the bottom mlp.
      mlp_top_dims: dimensions of dense layers of the top mlp.
      num_dense_features: number of dense features as the bottom mlp input.
      embed_dim: embedding dimension.
      keep_diags: whether to keep the diagonal terms in x @ x.T.
    """

    vocab_sizes: Sequence[int]
    total_vocab_sizes: int
    num_dense_features: int
    mlp_bottom_dims: Sequence[int] = (512, 256, 128)
    mlp_top_dims: Sequence[int] = (1024, 1024, 512, 256, 1)

    @nn.compact
    def __call__(self, x, train):
        del train
        embed_dim = 128

        bot_mlp_input, cat_features = jnp.split(x, [self.num_dense_features], 1)
        cat_features = jnp.asarray(cat_features, dtype=jnp.int32)

        # bottom mlp
        for dense_dim in self.mlp_bottom_dims:
            bot_mlp_input = nn.Dense(
                dense_dim,
                kernel_init=jnn.initializers.glorot_uniform(),
                bias_init=jnn.initializers.normal(stddev=jnp.sqrt(1.0 / dense_dim)),
            )(bot_mlp_input)
            bot_mlp_input = nn.relu(bot_mlp_input)
        bot_mlp_output = bot_mlp_input
        batch_size = bot_mlp_output.shape[0]
        feature_stack = jnp.reshape(bot_mlp_output, [batch_size, -1, embed_dim])

        # Embedding table look-up.
        vocab_sizes = jnp.asarray(self.vocab_sizes, dtype=jnp.int32)
        idx_offsets = jnp.asarray(
            [0] + list(jnp.cumsum(vocab_sizes[:-1])), dtype=jnp.int32
        )
        idx_offsets = jnp.tile(jnp.reshape(idx_offsets, [1, -1]), [batch_size, 1])
        idx_lookup = cat_features + idx_offsets
        # Scale the initialization to fan_in for each slice.
        scale = 1.0 / jnp.sqrt(vocab_sizes)
        scale = jnp.expand_dims(
            jnp.repeat(scale, vocab_sizes, total_repeat_length=self.total_vocab_sizes),
            -1,
        )

        def scaled_init(key, shape, scale, init, dtype=jnp.float_):
            return scale * init(key, shape, dtype)

        scaled_variance_scaling_init = functools.partial(
            scaled_init, scale=scale, init=jnn.initializers.uniform(scale=1.0)
        )
        embedding_table = self.param(
            "embedding_table",
            scaled_variance_scaling_init,
            [self.total_vocab_sizes, embed_dim],
        )

        idx_lookup = jnp.reshape(idx_lookup, [-1])
        embed_features = embedding_table[idx_lookup]
        embed_features = jnp.reshape(embed_features, [batch_size, -1, embed_dim])
        feature_stack = jnp.concatenate([feature_stack, embed_features], axis=1)
        dot_interact_output = dot_interact(concat_features=feature_stack)
        top_mlp_input = jnp.concatenate([bot_mlp_output, dot_interact_output], axis=-1)
        mlp_input_dim = top_mlp_input.shape[1]
        mlp_top_dims = self.mlp_top_dims
        num_layers_top = len(mlp_top_dims)
        for layer_idx, fan_out in enumerate(mlp_top_dims):
            fan_in = mlp_input_dim if layer_idx == 0 else mlp_top_dims[layer_idx - 1]
            top_mlp_input = nn.Dense(
                fan_out,
                kernel_init=jnn.initializers.normal(
                    stddev=jnp.sqrt(2.0 / (fan_in + fan_out))
                ),
                bias_init=jnn.initializers.normal(
                    stddev=jnp.sqrt(1.0 / mlp_top_dims[layer_idx])
                ),
            )(top_mlp_input)
            if layer_idx < (num_layers_top - 1):
                top_mlp_input = nn.relu(top_mlp_input)
        logits = top_mlp_input
        return logits
