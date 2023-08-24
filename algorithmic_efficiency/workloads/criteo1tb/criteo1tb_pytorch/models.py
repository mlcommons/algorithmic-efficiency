"""Pytorch implementation of DLRM-Small."""

import math

import torch
from torch import nn

from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


class DotInteract(nn.Module):
  """Performs feature interaction operation between dense or sparse features."""
  def __init__(self, num_sparse_features):
    super().__init__()
    self.triu_indices = torch.triu_indices(
      num_sparse_features + 1, num_sparse_features + 1
    )

  def forward(self, dense_features, sparse_features):
    combined_values = torch.cat(
      (dense_features.unsqueeze(1), sparse_features), dim=1
    )
    interactions = torch.bmm(
      combined_values, torch.transpose(combined_values, 1, 2)
    )
    interactions_flat = interactions[:, self.triu_indices[0], self.triu_indices[1]]
    return torch.cat((dense_features, interactions_flat), dim=1)


class DlrmSmall(nn.Module):
  """Define a DLRM-Small model.

  Parameters:
    vocab_size: vocab size of embedding table.
    num_dense_features: number of dense features as the bottom mlp input.
    mlp_bottom_dims: dimensions of dense layers of the bottom mlp.
    mlp_top_dims: dimensions of dense layers of the top mlp.
    embed_dim: embedding dimension.
  """

  def __init__(self,
               vocab_size,
               num_dense_features=13,
               num_sparse_features=26,
               mlp_bottom_dims=(512, 256, 128),
               mlp_top_dims=(1024, 1024, 512, 256, 1),
               embed_dim=128,
               dropout_rate=0.0):
    super().__init__()
    self.vocab_size = torch.tensor(vocab_size, dtype=torch.int32)
    self.num_dense_features = num_dense_features
    self.num_sparse_features = num_sparse_features
    self.mlp_bottom_dims = mlp_bottom_dims
    self.mlp_top_dims = mlp_top_dims
    self.embed_dim = embed_dim

    # Ideally, we should use the pooled embedding implementation from `TorchRec`.
    # However, in order to have identical implementation with that of Jax, we define a
    # single embedding matrix.
    num_chucks = 4
    assert vocab_size % num_chucks == 0
    # self.embedding_table_chucks = []
    scale = 1.0 / torch.sqrt(self.vocab_size)
    # self.embedding_table = nn.Parameter(torch.Tensor(self.vocab_size, self.embed_dim))
    for i in range(num_chucks):
      chunk = nn.Parameter(torch.Tensor(self.vocab_size // num_chucks, self.embed_dim))
      chunk.data.uniform_(0, 1)
      chunk.data = scale * chunk.data
      self.register_parameter(f"embedding_chunk_{i}", chunk)
      self.embedding_table_chucks.append(chunk)

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

    self.dot_interact = DotInteract(
      num_sparse_features=num_sparse_features,
    )

    input_dims = 506   # TODO: Write down the formula here instead of the constant.
    top_mlp_layers = []
    num_layers_top = len(self.mlp_top_dims)
    for layer_idx, fan_out in enumerate(self.mlp_top_dims):
      fan_in = input_dims if layer_idx == 0 \
          else self.mlp_top_dims[layer_idx - 1]
      top_mlp_layers.append(nn.Linear(fan_in, fan_out))
      if layer_idx < (num_layers_top - 1):
        top_mlp_layers.append(nn.ReLU(inplace=True))
      if (dropout_rate is not None and dropout_rate > 0.0 and
          layer_idx == num_layers_top - 2):
        top_mlp_layers.append(nn.Dropout(p=dropout_rate))
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
    batch_size = x.shape[0]

    dense_features, sparse_features = torch.split(
      x, [self.num_dense_features, self.num_sparse_features], 1)

    # Bottom MLP.
    embedded_dense = self.bot_mlp(dense_features)

    # Sparse feature processing.
    sparse_features = sparse_features.to(dtype=torch.int32)
    idx_lookup = torch.reshape(sparse_features, [-1]) % self.vocab_size
    embedding_table = torch.cat(self.embedding_table_chucks, dim=0)
    embedded_sparse = embedding_table[idx_lookup]
    # embedded_sparse = self.embedding_table[idx_lookup]
    embedded_sparse = torch.reshape(embedded_sparse,
                                    [batch_size, -1, self.embed_dim])

    # Dot product interactions.
    concatenated_dense = self.dot_interact(
      dense_features=embedded_dense, sparse_features=embedded_sparse
    )

    # Final MLP.
    logits = self.top_mlp(concatenated_dense)
    return logits
