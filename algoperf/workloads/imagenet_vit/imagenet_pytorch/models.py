"""PyTorch implementation of refactored and simplified ViT.

Adapted from:
https://github.com/huggingface/transformers/tree/main/src/transformers/models/vit
and https://github.com/lucidrains/vit-pytorch.
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from algoperf import init_utils
from algoperf import spec
from algoperf.workloads.wmt.wmt_pytorch.models import \
    MultiheadAttention


def posemb_sincos_2d(patches: spec.Tensor, temperature=10_000.) -> spec.Tensor:
  """Follows the MoCo v3 logic."""
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
      mlp_dim: Optional[int] = None,  # Defaults to 4x input dim.
      use_glu: bool = False,
      dropout_rate: float = 0.0) -> None:
    super().__init__()

    self.width = width
    self.mlp_dim = mlp_dim or 4 * width
    self.use_glu = use_glu
    self.dropout_rate = dropout_rate

    self.linear1 = nn.Linear(self.width, self.mlp_dim)
    self.act_fnc = nn.GELU(approximate='tanh')
    self.dropout = nn.Dropout(self.dropout_rate)

    if self.use_glu:
      self.glu_linear = nn.Linear(self.mlp_dim, self.mlp_dim)
    else:
      self.glu_linear = None

    self.linear2 = nn.Linear(self.mlp_dim, self.width)

    self.reset_parameters()

  def reset_parameters(self) -> None:
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
          module.bias.data.normal_(std=1e-6)

  def forward(self, x: spec.Tensor) -> spec.Tensor:
    x = self.linear1(x)
    x = self.act_fnc(x)

    if self.use_glu:
      y = self.glu_linear(x)
      x = x * y

    x = self.dropout(x)
    x = self.linear2(x)
    return x


class SelfAttention(nn.Module):
  """Self-attention special case of multi-head dot-product attention."""

  def __init__(self,
               width: int,
               num_heads: int = 8,
               dropout_rate: float = 0.0) -> None:
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
    self.dropout = nn.Dropout(dropout_rate)
    self.out = nn.Linear(self.width, self.width)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
          nn.init.constant_(module.bias.data, 0.)

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
               use_glu: bool = False,
               use_post_layer_norm: bool = False,
               dropout_rate: float = 0.0) -> None:
    super().__init__()

    self.width = width
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads
    self.use_glu = use_glu
    self.use_post_layer_norm = use_post_layer_norm

    self.layer_norm0 = nn.LayerNorm(self.width, eps=1e-6)
    self.self_attention1 = SelfAttention(self.width, self.num_heads)
    self.dropout = nn.Dropout(dropout_rate)
    self.layer_norm2 = nn.LayerNorm(self.width, eps=1e-6)
    self.mlp3 = MlpBlock(
        width=self.width,
        mlp_dim=self.mlp_dim,
        use_glu=self.use_glu,
        dropout_rate=dropout_rate)

  def forward(self, x: spec.Tensor) -> spec.Tensor:
    if not self.use_post_layer_norm:
      y = self.layer_norm0(x)
      y = self.self_attention1(y)
      y = self.dropout(y)
      x = x + y

      y = self.layer_norm2(x)
      y = self.mlp3(y)
      y = self.dropout(y)
      x = x + y
    else:
      y = x
      y = self.self_attention1(y)
      y = self.dropout(y)
      x = x + y
      x = self.layer_norm0(x)

      y = x
      y = self.mlp3(y)
      y = self.dropout(y)
      x = x + y
      x = self.layer_norm2(x)
    return x


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  def __init__(self,
               depth: int,
               width: int,
               mlp_dim: Optional[int] = None,
               num_heads: int = 12,
               use_glu: bool = False,
               use_post_layer_norm: bool = False,
               dropout_rate: float = 0.0) -> None:
    super().__init__()

    self.depth = depth
    self.width = width
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads
    self.use_glu = use_glu
    self.use_post_layer_norm = use_post_layer_norm

    self.net = nn.ModuleList([
        Encoder1DBlock(self.width,
                       self.mlp_dim,
                       self.num_heads,
                       self.use_glu,
                       self.use_post_layer_norm,
                       dropout_rate) for _ in range(depth)
    ])

    if not self.use_post_layer_norm:
      self.encoder_norm = nn.LayerNorm(self.width, eps=1e-6)
    else:
      self.encoder_norm = None

  def forward(self, x: spec.Tensor) -> spec.Tensor:
    # Input Encoder.
    for block in self.net:
      x = block(x)
    if not self.use_post_layer_norm:
      return self.encoder_norm(x)
    else:
      return x


class MAPHead(nn.Module):
  """Multihead Attention Pooling."""

  def __init__(self,
               width: int,
               mlp_dim: Optional[int] = None,
               num_heads: int = 12):
    super().__init__()
    self.width = width
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads

    self.probe = nn.Parameter(torch.zeros((1, 1, self.width)))
    nn.init.xavier_uniform_(self.probe.data)

    self.mha = MultiheadAttention(
        self.width, num_heads=self.num_heads, self_attn=False, bias=True)
    self.layer_norm = nn.LayerNorm(self.width, eps=1e-6)
    self.mlp = MlpBlock(width=self.width, mlp_dim=self.mlp_dim)

  def forward(self, x: spec.Tensor) -> spec.Tensor:
    n, _, _ = x.shape
    probe = torch.tile(self.probe, [n, 1, 1])

    x = self.mha(probe, x)[0]
    y = self.layer_norm(x)
    x = x + self.mlp(y)
    return x[:, 0]


class ViT(nn.Module):
  """ViT model."""

  image_height: int = 224
  image_width: int = 224
  channels: int = 3

  def __init__(
      self,
      num_classes: int = 1000,
      patch_size: Tuple[int, int] = (16, 16),
      width: int = 768,
      depth: int = 12,
      mlp_dim: Optional[int] = None,  # Defaults to 4x input dim.
      num_heads: int = 12,
      rep_size: Union[int, bool] = True,
      dropout_rate: Optional[float] = 0.0,
      head_zeroinit: bool = True,
      use_glu: bool = False,
      use_post_layer_norm: bool = False,
      use_map: bool = False,
      dtype: Any = torch.float32) -> None:
    super().__init__()
    if dropout_rate is None:
      dropout_rate = 0.0

    self.num_classes = num_classes
    self.patch_size = patch_size
    self.width = width
    self.depth = depth
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads
    self.rep_size = rep_size
    self.head_zeroinit = head_zeroinit
    self.use_glu = use_glu
    self.use_post_layer_norm = use_post_layer_norm
    self.use_map = use_map
    self.dtype = dtype

    if self.rep_size:
      rep_size = self.width if self.rep_size is True else self.rep_size
      self.pre_logits = nn.Linear(self.width, rep_size)

    self.conv_patch_extract = nn.Conv2d(
        self.channels,
        self.width,
        self.patch_size,
        stride=self.patch_size,
        padding='valid')
    self.dropout = nn.Dropout(p=dropout_rate)

    self.encoder = Encoder(
        depth=self.depth,
        width=self.width,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        use_glu=self.use_glu,
        use_post_layer_norm=self.use_post_layer_norm,
        dropout_rate=dropout_rate)

    if self.num_classes:
      self.head = nn.Linear(self.width, self.num_classes)

    if self.use_map:
      self.map = MAPHead(self.width, self.mlp_dim, self.num_heads)
    else:
      self.map = None

    self.reset_parameters()

  def reset_parameters(self) -> None:
    init_utils.pytorch_default_init(self.conv_patch_extract)

    if self.rep_size:
      init_utils.pytorch_default_init(self.pre_logits)

    if self.num_classes:
      if self.head_zeroinit:
        nn.init.constant_(self.head.weight.data, 0.)
        nn.init.constant_(self.head.bias.data, 0.)
      else:
        init_utils.pytorch_default_init(self.head)

  def get_posemb(self, x: spec.Tensor) -> spec.Tensor:
    return posemb_sincos_2d(x).type(self.dtype)

  def forward(self, x: spec.Tensor) -> spec.Tensor:
    # Patch extraction.
    x = self.conv_patch_extract(x)

    # Add posemb before adding extra token.
    n, c, h, w = x.shape
    pes = self.get_posemb(x)

    # Reshape to match Jax's ViT implementation.
    x = torch.transpose(torch.reshape(x, (n, c, h * w)), 1, 2)
    x = x + pes

    x = self.dropout(x)
    x = self.encoder(x)

    if self.use_map:
      x = self.map(x)
    else:
      x = torch.mean(x, dim=1)

    if self.rep_size:
      x = torch.tanh(self.pre_logits(x))

    if self.num_classes:
      x = self.head(x)

    return x
