"""A refactored and simplified ViT.

Note: Adapted from
https://github.com/huggingface/transformers/tree/main/src/transformers/models/vit
https://github.com/lucidrains/vit-pytorch
"""

import math
from typing import Any, Optional, Tuple, Union
from algorithmic_efficiency import spec
from algorithmic_efficiency import init_utils
import torch
from torch import nn
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


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  def __init__(
      self,
      width: int,
      mlp_dim: Optional[int] = None,  # Defaults to 4x input dim
      dropout_rate: float = 0.0) -> None:
    super().__init__()

    self.width = width
    self.mlp_dim = mlp_dim or 4 * width
    self.dropout_rate = dropout_rate

    self.net = nn.Sequential(
        nn.Linear(self.width, self.mlp_dim),
        nn.GELU(),
        nn.Dropout(self.dropout_rate),
        nn.Linear(self.mlp_dim, self.width))
    self.reset_parameters()

  def reset_parameters(self) -> None:
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          module.bias.data.normal_(std=1e-6)

  def forward(self, x: spec.Tensor) -> spec.Tensor:
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
    self.key = nn.Linear(self.width, self.all_head_dim)
    self.value = nn.Linear(self.width, self.all_head_dim)

    self.dropout = nn.Dropout(dropout)
    self.out = nn.Linear(self.width, self.width)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.constant_(module.bias, 0.)

  def transpose_for_scores(self, x: spec.Tensor) -> spec.Tensor:
    new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, x: spec.Tensor) -> spec.Tensor:
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
               dropout_rate: float = 0.0) -> None:
    super().__init__()

    self.width = width
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads

    self.layer_norm0 = nn.LayerNorm(self.width)
    self.self_attention1 = SelfAttention(self.width, self.num_heads)
    self.dropout = nn.Dropout(dropout_rate)
    self.layer_norm2 = nn.LayerNorm(self.width)
    self.mlp3 = MlpBlock(self.width, self.mlp_dim, dropout_rate)

  def forward(self, x: spec.Tensor) -> spec.Tensor:
    y = self.layer_norm0(x)
    y = self.self_attention1(y)
    y = self.dropout(y)
    x = x + y

    y = self.layer_norm2(x)
    y = self.mlp3(y)
    y = self.dropout(y)
    x = x + y
    return x


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

  def forward(self, x: spec.Tensor) -> spec.Tensor:
    for lyr, block in enumerate(self.net):
      x = block(x)
    return self.encoder_norm(x)


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
      self.pos_embed = nn.Parameter(torch.randn(1, num_patches, width))
      nn.init.normal_(self.pos_embed.data, std=1 / math.sqrt(self.width))

    if self.pool_type == 'tok':
      self.cls = nn.Parameter(torch.randn(1, 1, width)).type(self.dtype)
      nn.init.constant_(self.cls.data, 0.)

    if self.rep_size:
      rep_size = self.width if self.rep_size is True else self.rep_size  # pylint: disable=g-bool-id-comparison
      self.pre_logits = nn.Linear(self.width, rep_size)

    self.embed = nn.Conv2d(
        self.channels,
        self.width,
        self.patch_size,
        stride=self.patch_size,
        padding='valid')
    self.dropout = nn.Dropout(p=dropout)

    self.encoder = Encoder(
        depth=self.depth,
        width=self.width,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=dropout)

    if self.num_classes:
      self.head = nn.Linear(self.width, self.num_classes)
    if self.head_zeroinit:
      self.head.weight.data.zero_()

  def reset_parameters(self):
    init_utils.pytorch_default_init(self.embed)

    if self.rep_size:
      init_utils.pytorch_default_init(self.pre_logits)

    if self.num_classes:
      if self.head_zeroinit:
        nn.init.constant_(self.head.weight.data, 0.)
        nn.init.constant_(self.head.bias.data, 0.)
      else:
        init_utils.pytorch_default_init(self.pre_logits)

  def get_posemb(self, x: spec.Tensor) -> spec.Tensor:
    if self.posemb == 'learn':
      return self.pos_embed.type(self.dtype)
    elif self.posemb == 'sincos2d':
      return posemb_sincos_2d(x).type(self.dtype)
    else:
      raise ValueError(f'Unknown posemb type: {self.posemb}')

  def forward(self, x: spec.Tensor) -> spec.Tensor:
    # Patch extraction
    x = self.embed(x)

    # Add posemb before adding extra token.
    n, c, h, w = x.shape
    pes = self.get_posemb(x)

    # Reshape to match Jax's ViT implementation.
    x = torch.transpose(torch.reshape(x, (n, c, h * w)), 1, 2)
    x = x + pes

    if self.pool_type == 'tok':
      x = torch.cat((torch.tile(self.cls, [n, 1, 1]), x), dim=1)

    x = self.dropout(x)
    x = self.encoder(x)
    if self.pool_type == 'gap':
      x = torch.mean(x, dim=1)
    elif self.pool_type == '0':
      x = x[:, 0]
    elif self.pool_type == 'tok':
      x = x[:, 0]
    else:
      raise ValueError(f'Unknown pool type: "{self.pool_type}"')

    if self.rep_size:
      x = torch.tanh(self.pre_logits(x))

    if self.num_classes:
      x = self.head(x)
    return x
