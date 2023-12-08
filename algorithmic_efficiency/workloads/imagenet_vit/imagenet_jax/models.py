"""Jax implementation of refactored and simplified ViT.

Forked from:
https://github.com/google/init2winit/blob/master/init2winit/model_lib/vit.py,
originally from https://github.com/google/big_vision with modifications noted.
"""

from typing import Optional, Sequence, Union

from flax import linen as nn
import jax.numpy as jnp

from algorithmic_efficiency import spec


def posemb_sincos_2d(h: int,
                     w: int,
                     width: int,
                     temperature: int = 10_000.,
                     dtype: jnp.dtype = jnp.float32) -> spec.Tensor:
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]  #pylint: disable=unpacking-non-sequence

  if width % 4 != 0:
    raise ValueError('Width must be mult of 4 for sincos posemb.')
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1. / (temperature**omega)
  y = jnp.einsum('m,d->md', y.flatten(), omega)
  x = jnp.einsum('m,d->md', x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim.
  use_glu: bool = False
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x: spec.Tensor, train: bool = True) -> spec.Tensor:
    """Applies Transformer MlpBlock module."""
    inits = {
        'kernel_init': nn.initializers.xavier_uniform(),
        'bias_init': nn.initializers.normal(stddev=1e-6),
    }

    d = x.shape[2]
    x = nn.Dense(self.mlp_dim or 4 * d, **inits)(x)
    x = nn.gelu(x)

    if self.use_glu:
      y = nn.Dense(
        self.mlp_dim,
        **inits)(x)
      x = x * y

    x = nn.Dropout(rate=self.dropout_rate)(x, train)
    x = nn.Dense(d, **inits)(x)
    return x


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim.
  num_heads: int = 12
  use_glu: bool = False
  use_post_layer_norm: bool = False
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x: spec.Tensor, train: bool = True) -> spec.Tensor:
    if not self.use_post_layer_norm:
        y = nn.LayerNorm(name='LayerNorm_0')(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=train,
            name='MultiHeadDotProductAttention_1')(
                y)
        y = nn.Dropout(rate=self.dropout_rate)(y, train)
        x = x + y

        y = nn.LayerNorm(name='LayerNorm_2')(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim, use_glu=self.use_glu, dropout_rate=self.dropout_rate,
            name='MlpBlock_3')(y, train)
        y = nn.Dropout(rate=self.dropout_rate)(y, train)
        x = x + y
    else:
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=train,
            name='MultiHeadDotProductAttention_1')(
            x)
        y = nn.Dropout(rate=self.dropout_rate)(y, train)
        x = x + y
        x = nn.LayerNorm(name='LayerNorm_0')(x)

        y = MlpBlock(
            mlp_dim=self.mlp_dim, use_glu=self.use_glu, dropout_rate=self.dropout_rate,
            name='MlpBlock_3')(x, train)
        y = nn.Dropout(rate=self.dropout_rate)(y, train)
        x = x + y
        x = nn.LayerNorm(name='LayerNorm_2')(x)

    return x


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  depth: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim.
  num_heads: int = 12
  dropout_rate: float = 0.0
  use_glu: bool = False
  use_post_layer_norm: bool = False

  @nn.compact
  def __call__(self, x: spec.Tensor, train: bool = True) -> spec.Tensor:
    # Input Encoder
    for lyr in range(self.depth):
      block = Encoder1DBlock(
          name=f'encoderblock_{lyr}',
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          use_glu=self.use_glu,
          use_post_layer_norm=self.use_post_layer_norm,
          dropout_rate=self.dropout_rate)
      x = block(x, train)
    if not self.use_post_layer_norm:
      return nn.LayerNorm(name='encoder_layernorm')(x)
    else:
      return x


class MAPHead(nn.Module):
  """Multihead Attention Pooling."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  @nn.compact
  def __call__(self, x):
    n, _, d = x.shape
    probe = self.param('probe',
                       nn.initializers.xavier_uniform(),
                       (1, 1, d), x.dtype)
    probe = jnp.tile(probe, [n, 1, 1])

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform())(probe, x)

    y = nn.LayerNorm()(x)
    x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
    return x[:, 0]


class ViT(nn.Module):
  """ViT model."""

  num_classes: int = 1000
  patch_size: Sequence[int] = (16, 16)
  width: int = 768
  depth: int = 12
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim.
  num_heads: int = 12
  rep_size: Union[int, bool] = True
  dropout_rate: Optional[float] = 0.0  # If None, defaults to 0.0.
  reinit: Optional[Sequence[str]] = None
  head_zeroinit: bool = True
  use_glu: bool = False,
  use_post_layer_norm: bool = False,
  use_map: bool = False,

  def get_posemb(self,
                 seqshape: tuple,
                 width: int,
                 dtype: jnp.dtype = jnp.float32) -> spec.Tensor:
    return posemb_sincos_2d(*seqshape, width, dtype=dtype)

  @nn.compact
  def __call__(self, x: spec.Tensor, *, train: bool = False) -> spec.Tensor:
    # Patch extraction
    x = nn.Conv(
        self.width,
        self.patch_size,
        strides=self.patch_size,
        padding='VALID',
        name='conv_patch_extract')(
            x)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # Add posemb before adding extra token.
    x = x + self.get_posemb((h, w), c, x.dtype)

    dropout_rate = self.dropout_rate
    if dropout_rate is None:
      dropout_rate = 0.0
    x = nn.Dropout(rate=dropout_rate)(x, not train)

    x = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        use_glu=self.use_glu,
        use_post_layer_norm=self.use_post_layer_norm,
        dropout_rate=dropout_rate,
        name='Transformer')(
            x, train=not train)

    if self.use_map:
      x = MAPHead(num_heads=self.num_heads,
                  mlp_dim=self.mlp_dim
                  )(x)
    else:
      x = jnp.mean(x, axis=1)

    if self.rep_size:
      rep_size = self.width if self.rep_size is True else self.rep_size
      hid = nn.Dense(rep_size, name='pre_logits')
      x = nn.tanh(hid(x))

    if self.num_classes:
      kw = {'kernel_init': nn.initializers.zeros} if self.head_zeroinit else {}
      head = nn.Dense(self.num_classes, name='head', **kw)
      x = head(x)

    return x
