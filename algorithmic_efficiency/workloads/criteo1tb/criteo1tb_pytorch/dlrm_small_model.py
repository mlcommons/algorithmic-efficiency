"""Pytorch implementation of DLRM -Small.
Data parallel and smaller embedding tables compared to the MLPerf training implementation.
"""

import functools
import math
from typing import Sequence

from absl import logging
import numpy as np
import torch
import torch.nn as nn


def dot_interact(concat_features, device):
  """ Performs feature interaction operation between desnse and sparse features.
    Inpute tensors represent dense or sparse features.
    """
  batch_size = concat_features.shape[0]
  xactions = torch.bmm(concat_features,
                       torch.permute(concat_features, (0, 2, 1)))
  feature_dim = xactions.shape[-1]
  indices = torch.triu_indices(feature_dim, feature_dim, device=device)
  num_elems = indices.shape[1]
  indices = torch.tile(indices, [1, batch_size])
  indices0 = torch.reshape(
      torch.tile(
          torch.reshape(torch.arange(batch_size, device=device), [-1, 1]),
          [1, num_elems]), [1, -1])
  indices = tuple(torch.cat((indices0, indices), 0))
  activations = xactions[indices]
  activations = torch.reshape(activations, [batch_size, -1])
  return activations


class DlrmSmall(nn.Module):
  """
    Define a DLRM-Small model.

    device: GPU OR CPU. It is used here because some tensor/ops start from device
    vocab_size: list of vocab sizes of embedding tables.
    total_vocab_sizes: sum of embedding table sizes
    mlp_bottom_dims: dims of dense layers of the bottom mlp.
    mlp_top_dims: dims of dense laters of the top mlp. 
    num_dense_features: number of dense features as the bottom mlp input.
    embded_dim: embedding dimension.
    """

  def __init__(self,
               vocab_sizes,
               total_vocab_sizes,
               num_dense_features=13,
               num_sparse_features=26,
               mlp_bottom_dims=(512, 256, 128),
               mlp_top_dims=(1024, 1024, 512, 256, 1),
               embed_dim=128,
               device="cuda"):
    super(DlrmSmall, self).__init__()
    # Embedding table lookup
    self.device = device
    self.vocab_sizes = torch.asarray(
        vocab_sizes, dtype=torch.int32, device=self.device)
    self.total_vocab_sizes = total_vocab_sizes
    self.num_dense_features = num_dense_features
    self.num_sparse_features = num_sparse_features
    self.mlp_bottom_dims = mlp_bottom_dims
    self.mlp_top_dims = mlp_top_dims
    self.embed_dim = embed_dim
    self.device = device
    self.embedding_table = nn.Embedding(
        self.total_vocab_sizes, self.embed_dim, device=self.device)
    self.embedding_table.weight.data.uniform_(0, 1)
    # Scale the initialization to fan_in for each slice.
    scale = 1.0 / torch.sqrt(self.vocab_sizes)
    scale = torch.unsqueeze(
        torch.repeat_interleave(
            scale, self.vocab_sizes, output_size=self.total_vocab_sizes),
        dim=-1)
    self.embedding_table.weight.data = scale * self.embedding_table.weight.data
    bottom_mlp_layers = []
    input_dim = self.num_dense_features

    for dense_dim in self.mlp_bottom_dims:
      bottom_mlp_layers.append(nn.Linear(input_dim, dense_dim))
      bottom_mlp_layers.append(nn.ReLU(inplace=True))
      input_dim = dense_dim
    self.bot_mlp = nn.Sequential(*bottom_mlp_layers).to(device)

    for module in self.bot_mlp.modules():
      if isinstance(module, nn.Linear):
        nn.init.normal_(
            module.weight.data,
            0.,
            math.sqrt(2. / (module.in_features + module.out_features)))
        nn.init.normal_(module.bias.data,
                        0.,
                        math.sqrt(1. / module.out_features))

    # precomputed dot interaction self input_dim
    # number of sparse features = 26
    input_dims = ((self.num_sparse_features + 1) *
                  (self.num_sparse_features + 1 + 1)) // 2 + self.embed_dim
    top_mlp_layers = []

    for output_dims in self.mlp_top_dims[:-1]:
      top_mlp_layers.append(nn.Linear(input_dims, output_dims))
      top_mlp_layers.append(nn.ReLU(inplace=True))
      input_dims = output_dims
    top_mlp_layers.append(nn.Linear(input_dims, self.mlp_top_dims[-1]))
    self.top_mlp = nn.Sequential(*top_mlp_layers).to(self.device)

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
    bot_mlp_input, cat_features = x
    cat_features = cat_features.to(dtype=torch.int32)
    bot_mlp_output = self.bot_mlp(bot_mlp_input)
    batch_size = bot_mlp_output.shape[0]
    feature_stack = torch.reshape(bot_mlp_output,
                                  [batch_size, -1, self.embed_dim])
    idx_offsets = torch.asarray(
        [0] + list(torch.cumsum(vocab_sizes[:-1], dim=0)),
        dtype=torch.int32,
        device=self.device)
    idx_offsets = torch.tile(
        torch.reshape(idx_offsets, [1, -1]), [batch_size, 1])
    idx_lookup = cat_features + idx_offsets
    idx_lookup = torch.reshape(idx_lookup, [-1])
    embed_features = self.embedding_table(idx_lookup)
    embed_features = torch.reshape(embed_features,
                                   [batch_size, -1, self.embed_dim])
    feature_stack = torch.cat([feature_stack, embed_features], axis=1)
    dot_interact_output = dot_interact(
        concat_features=feature_stack, device=self.device)
    top_mlp_input = torch.cat([bot_mlp_output, dot_interact_output], axis=-1)
    logits = self.top_mlp(top_mlp_input)
    return logits
