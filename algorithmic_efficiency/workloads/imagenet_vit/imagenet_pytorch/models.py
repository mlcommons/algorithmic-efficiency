"""A refactored and simplified ViT.

Adapted from:
https://github.com/huggingface/transformers/tree/main/src/transformers/models/vit
https://github.com/lucidrains/vit-pytorch
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


def posemb_sincos_2d(patches, temperature=10_000.):
  _, width, h, w = patches.shape
  device = patches.device
  y, x = torch.meshgrid(torch.arange(h, device=device),
                        torch.arange(w, device=device), indexing='ij')

  if width % 4 != 0:
    raise ValueError('Width must be mult of 4 for sincos posemb.')
  omega = torch.arange(width // 4, device=device) / (width // 4 - 1)
  omega = 1. / (temperature**omega)
  y = y.flatten()[:, None] * omega[None, :]
  x = x.flatten()[:, None] * omega[None, :]
  pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
  return pe[None, :, :]


def init_weights(module) -> None:
  if isinstance(module, (nn.Linear, nn.Conv2d)):
    # Slightly different from the TF version which uses truncated_normal
    # for initialization, cf https://github.com/pytorch/pytorch/pull/5617
    module.weight.data.normal_(mean=0.0, std=0.02)
    if module.bias is not None:
      module.bias.data.zero_()
  elif isinstance(module, nn.LayerNorm):
    module.bias.data.zero_()
    module.weight.data.fill_(1.0)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  def __init__(
      self,
      width: int,
      mlp_dim: Optional[int] = None,  # Defaults to 4x input dim
      dropout: float = 0.0) -> None:
    super().__init__()

    self.width = width
    self.mlp_dim = mlp_dim or 4 * width
    self.dropout = dropout

    self.net = nn.Sequential(
        nn.Linear(self.width, self.mlp_dim),
        nn.GELU(),
        nn.Dropout(self.dropout),
        nn.Linear(self.mlp_dim, self.width))
    self._init_weights()

  def _init_weights(self) -> None:
    for module in self.modules():
      if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          module.bias.data.normal_(std=1e-6)

  def forward(self, x: Tensor) -> Tensor:
    return self.net(x)


class SelfAttention(nn.Module):
  """Self-attention special case of multi-head dot-product attention."""

  def __init__(self,
               width: int,
               num_heads: int = 8,
               dropout: float = 0.0) -> None:
    super().__init__()

    self.width = width
    self.num_heads = num_heads

    assert width % num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')

    self.head_dim = int(width / num_heads)
    self.all_head_dim = self.num_heads * self.head_dim

    self.query = nn.Linear(self.width, self.all_head_dim)
    torch.nn.init.xavier_uniform_(self.query.weight)
    self.query.bias.data.zero_()
    self.key = nn.Linear(self.width, self.all_head_dim)
    torch.nn.init.xavier_uniform_(self.key.weight)
    self.key.bias.data.zero_()
    self.value = nn.Linear(self.width, self.all_head_dim)
    torch.nn.init.xavier_uniform_(self.value.weight)
    self.value.bias.data.zero_()

    self.dropout = nn.Dropout(dropout)
    self.out = nn.Linear(self.width, self.width)
    torch.nn.init.xavier_uniform_(self.out.weight)
    self.out.bias.data.zero_()

  def transpose_for_scores(self, x: Tensor) -> Tensor:
    new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, x: Tensor) -> Tensor:
    mixed_query_layer = self.query(x)

    key_layer = self.transpose_for_scores(self.key(x))
    value_layer = self.transpose_for_scores(self.value(x))
    query_layer = self.transpose_for_scores(mixed_query_layer)

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.head_dim)

    attention_probs = F.softmax(attention_scores, dim=-1)
    attention_probs = self.dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_dim,)
    context_layer = context_layer.view(new_context_layer_shape)
    out = self.out(context_layer)
    return out


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""

  def __init__(self,
               width: int,
               mlp_dim: Optional[int] = None,
               num_heads: int = 12,
               dropout: float = 0.0) -> None:
    super().__init__()

    self.width = width
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads

    self.layer_norm0 = nn.LayerNorm(self.width)
    init_weights(self.layer_norm0)
    self.self_attention1 = SelfAttention(self.width, self.num_heads)
    self.dropout = nn.Dropout(dropout)
    self.layer_norm2 = nn.LayerNorm(self.width)
    init_weights(self.layer_norm2)
    self.mlp3 = MlpBlock(self.width, self.mlp_dim, dropout)

  def forward(self, x: Tensor) -> Tuple[Tensor, dict]:
    out = {}
    y = self.layer_norm0(x)
    y = out['sa'] = self.self_attention1(y)
    y = self.dropout(y)
    x = out['+sa'] = x + y

    y = self.layer_norm2(x)
    y = out['mlp'] = self.mlp3(y)
    y = self.dropout(y)
    x = out['+mlp'] = x + y
    return x, out


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  def __init__(self,
               depth: int,
               width: int,
               mlp_dim: Optional[int] = None,
               num_heads: int = 12,
               dropout: float = 0.0) -> None:
    super().__init__()

    self.depth = depth
    self.width = width
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads

    self.net = nn.ModuleList([
        Encoder1DBlock(self.width, self.mlp_dim, self.num_heads, dropout)
        for _ in range(depth)
    ])
    self.encoder_norm = nn.LayerNorm(self.width)
    init_weights(self.encoder_norm)

  def forward(self, x: Tensor) -> Tuple[Tensor, dict]:
    out = {}

    # Input Encoder
    for lyr, block in enumerate(self.net):
      x, out[f'block{lyr:02d}'] = block(x)
    out['pre_ln'] = x  # Alias for last block, but without the number in it.

    return self.encoder_norm(x), out


class ViT(nn.Module):
  """ViT model."""

  image_height: int = 224
  image_width: int = 224
  channels: int = 3

  def __init__(
      self,
      num_classes: int = 1000,
      patch_size: Tuple[int] = (16, 16),
      width: int = 768,
      depth: int = 12,
      mlp_dim: Optional[int] = None,  # Defaults to 4x input dim
      num_heads: int = 12,
      posemb: str = 'sincos2d',  # Can also be 'learn'
      rep_size: Union[int, bool] = True,
      dropout: float = 0.0,
      pool_type: str = 'gap',  # Can also be 'tok'
      head_zeroinit: bool = True,
      dtype: Any = torch.float32) -> None:
    super().__init__()

    self.num_classes = num_classes
    self.patch_size = patch_size
    self.width = width
    self.depth = depth
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads
    self.posemb = posemb
    self.rep_size = rep_size
    self.pool_type = pool_type
    self.head_zeroinit = head_zeroinit
    self.dtype = dtype

    num_patches = (self.image_height // self.patch_size[0]) * \
                  (self.image_width // self.patch_size[1])
    if self.posemb == 'learn':
      self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, width))
      self.pos_embedding.data.normal_(std=1 / math.sqrt(self.width))

    if self.pool_type == 'tok':
      self.cls = nn.Parameter(torch.randn(1, 1, width)).type(self.dtype)
      self.cls.data.zero_()

    if self.rep_size:
      rep_size = self.width if self.rep_size is True else self.rep_size  # pylint: disable=g-bool-id-comparison
      self.pre_logits = nn.Linear(self.width, rep_size)
      init_weights(self.pre_logits)

    self.embedding = nn.Conv2d(
        self.channels,
        self.width,
        self.patch_size,
        stride=self.patch_size,
        padding='valid')
    init_weights(self.embedding)
    self.dropout = nn.Dropout(p=dropout)

    self.encoder = Encoder(
        depth=self.depth,
        width=self.width,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=dropout)

    if self.num_classes:
      self.head = nn.Linear(self.width, self.num_classes)
      init_weights(self.head)
    if self.head_zeroinit:
      self.head.weight.data.zero_()

  def get_posemb(self, x: Tensor) -> Tensor:
    if self.posemb == 'learn':
      return self.pos_embedding.type(self.dtype)
    elif self.posemb == 'sincos2d':
      return posemb_sincos_2d(x).type(self.dtype)
    else:
      raise ValueError(f'Unknown posemb type: {self.posemb}')

  def forward(self, x: Tensor) -> Tensor:
    out = {}

    # Patch extraction
    x = out['stem'] = self.embedding(x)

    # Add posemb before adding extra token.
    n, c, h, w = x.shape
    pes = self.get_posemb(x)

    # Reshape to match Jax's ViT implementation.
    x = torch.transpose(torch.reshape(x, (n, c, h * w)), 1, 2)
    x = out['with_posemb'] = x + pes

    if self.pool_type == 'tok':
      x = torch.cat((torch.tile(self.cls, [n, 1, 1]), x), dim=1)

    n, l, c = x.shape  # pylint: disable=unused-variable
    x = self.dropout(x)

    x, out['encoder'] = self.encoder(x)
    encoded = out['encoded'] = x

    if self.pool_type == 'gap':
      x = out['head_input'] = torch.mean(x, dim=1)
    elif self.pool_type == '0':
      x = out['head_input'] = x[:, 0]
    elif self.pool_type == 'tok':
      x = out['head_input'] = x[:, 0]
      encoded = encoded[:, 1:]
    else:
      raise ValueError(f'Unknown pool type: "{self.pool_type}"')

    x_2d = torch.reshape(encoded, [n, h, w, -1])

    if self.rep_size:
      x_2d = torch.tanh(self.pre_logits(x_2d))
      x = torch.tanh(self.pre_logits(x))

    out['pre_logits_2d'] = x_2d
    out['pre_logits'] = x

    if self.num_classes:
      x_2d = out['logits_2d'] = self.head(x_2d)
      x = out['logits'] = self.head(x)
    return x
