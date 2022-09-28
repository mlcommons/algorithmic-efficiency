"""Pytorch implementation of DLRM-Small."""

import math

import torch
import torch.nn as nn


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
  xactions = torch.bmm(concat_features,
                       torch.permute(concat_features, (0, 2, 1)))
  feature_dim = xactions.shape[-1]

  indices = torch.triu_indices(feature_dim, feature_dim)
  num_elems = indices.shape[1]
  indices = torch.tile(indices, [1, batch_size])
  indices0 = torch.reshape(
      torch.tile(
          torch.reshape(torch.arange(batch_size), [-1, 1]), [1, num_elems]),
      [1, -1])
  indices = tuple(torch.cat((indices0, indices), 0))
  activations = xactions[indices]
  activations = torch.reshape(activations, [batch_size, -1])
  return activations


class DlrmSmall(nn.Module):
  """Define a DLRM-Small model.

  Parameters:
    vocab_sizes: list of vocab sizes of embedding tables.
    total_vocab_sizes: sum of embedding table sizes (for jit compilation).
    num_dense_features: number of dense features as the bottom mlp input.
    mlp_bottom_dims: dimensions of dense layers of the bottom mlp.
    mlp_top_dims: dimensions of dense layers of the top mlp.
    embed_dim: embedding dimension.
  """

  def __init__(self,
               vocab_sizes,
               total_vocab_sizes,
               num_dense_features=13,
               num_sparse_features=26,
               mlp_bottom_dims=(512, 256, 128),
               mlp_top_dims=(1024, 1024, 512, 256, 1),
               embed_dim=128):
    super(DlrmSmall, self).__init__()
    self.vocab_sizes = torch.tensor(vocab_sizes, dtype=torch.int32)
    self.total_vocab_sizes = total_vocab_sizes
    self.num_dense_features = num_dense_features
    self.num_sparse_features = num_sparse_features
    self.mlp_bottom_dims = mlp_bottom_dims
    self.mlp_top_dims = mlp_top_dims
    self.embed_dim = embed_dim

    self.register_buffer(
        "idx_offsets",
        torch.tensor(
            [0] + list(torch.cumsum(self.vocab_sizes[:-1], dim=0)),
            dtype=torch.int32))
    self.embedding_table = nn.Embedding(self.total_vocab_sizes, self.embed_dim)
    self.embedding_table.weight.data.uniform_(0, 1)
    # Scale the initialization to fan_in for each slice.
    scale = 1.0 / torch.sqrt(self.vocab_sizes)
    scale = torch.unsqueeze(
        torch.repeat_interleave(
            scale, self.vocab_sizes, output_size=self.total_vocab_sizes),
        dim=-1)
    self.embedding_table.weight.data = scale * self.embedding_table.weight.data

    # bottom mlp
    bottom_mlp_layers = []
    input_dim = self.num_dense_features
    for dense_dim in self.mlp_bottom_dims:
      bottom_mlp_layers.append(nn.Linear(input_dim, dense_dim))
      bottom_mlp_layers.append(nn.ReLU(inplace=True))
      input_dim = dense_dim
    self.bot_mlp = nn.Sequential(*bottom_mlp_layers)
    for module in self.bot_mlp.modules():
      if isinstance(module, nn.Linear):
        limit = math.sqrt(6. / (module.in_features + module.out_features))
        nn.init.uniform_(module.weight.data, -limit, limit)
        nn.init.normal_(module.bias.data,
                        0.,
                        math.sqrt(1. / module.out_features))

    # top mlp
    # precomputed dot interaction self input_dim
    # number of sparse features = 26
    input_dims = ((self.num_sparse_features + 1) *
                  (self.num_sparse_features + 1 + 1)) // 2 + self.embed_dim
    top_mlp_layers = []
    num_layers_top = len(self.mlp_top_dims)
    for layer_idx, fan_out in enumerate(self.mlp_top_dims):
      fan_in = input_dims if layer_idx == 0 \
          else self.mlp_top_dims[layer_idx - 1]
      top_mlp_layers.append(nn.Linear(fan_in, fan_out))
      if layer_idx < (num_layers_top - 1):
        top_mlp_layers.append(nn.ReLU(inplace=True))
    self.top_mlp = nn.Sequential(*top_mlp_layers)
    for module in self.top_mlp.modules():
      if isinstance(module, nn.Linear):
        nn.init.normal_(
            module.weight.data,
            0.,
            math.sqrt(2. / (module.in_features + module.out_features)))
        nn.init.normal_(module.bias.data,
                        0.,
                        math.sqrt(1. / module.out_features))

  def forward(self, x):
    print(x)
    bot_mlp_input, cat_features = torch.split(x, [self.num_dense_features], 1)
    cat_features = cat_features.to(dtype=torch.int32)
    bot_mlp_output = self.bot_mlp(bot_mlp_input)
    batch_size = bot_mlp_output.shape[0]
    feature_stack = torch.reshape(bot_mlp_output,
                                  [batch_size, -1, self.embed_dim])
    idx_offsets = torch.tile(
        torch.reshape(self.idx_offsets, [1, -1]), [batch_size, 1])
    idx_lookup = cat_features + idx_offsets
    idx_lookup = torch.reshape(idx_lookup, [-1])
    embed_features = self.embedding_table(idx_lookup)
    embed_features = torch.reshape(embed_features,
                                   [batch_size, -1, self.embed_dim])
    feature_stack = torch.cat([feature_stack, embed_features], axis=1)
    dot_interact_output = dot_interact(concat_features=feature_stack)
    top_mlp_input = torch.cat([bot_mlp_output, dot_interact_output], axis=-1)
    logits = self.top_mlp(top_mlp_input)
    return logits
