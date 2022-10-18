"""A refactored and simplified ViT.

NOTE: Forked from
https://github.com/google/init2winit/blob/master/init2winit/model_lib/vit.py,
originally from https://github.com/google/big_vision with modifications noted.
"""

from typing import Optional, Sequence, Union

from flax import linen as nn
import jax.numpy as jnp
import numpy as np

from algorithmic_efficiency import spec


def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32):
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]

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
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x: spec.Tensor, train: bool = True) -> spec.Tensor:
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )

    d = x.shape[2]
    x = nn.Dense(self.mlp_dim or 4 * d, **inits)(x)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, train)
    x = nn.Dense(d, **inits)(x)
    return x


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x: spec.Tensor, train: bool = True) -> spec.Tensor:
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
        mlp_dim=self.mlp_dim, dropout_rate=self.dropout_rate,
        name='MlpBlock_3')(y, train)
    y = nn.Dropout(rate=self.dropout_rate)(y, train)
    x = x + y
    return x


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  depth: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x: spec.Tensor, train: bool = True) -> spec.Tensor:
    # Input Encoder
    for lyr in range(self.depth):
      block = Encoder1DBlock(
          name=f'encoderblock_{lyr}',
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate)
      x = block(x, train)
    return nn.LayerNorm(name='encoder_norm')(x)


class ViT(nn.Module):
  """ViT model."""

  num_classes: int = 1000
  patch_size: Sequence[int] = (16, 16)
  width: int = 768
  depth: int = 12
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  rep_size: Union[int, bool] = True
  dropout_rate: Optional[float] = 0.0  # If None, defaults to 0.0.
  reinit: Optional[Sequence[str]] = None
  head_zeroinit: bool = True

  def get_posemb(self, seqshape, width, dtype=jnp.float32):
    return posemb_sincos_2d(*seqshape, width, dtype=dtype)

  @nn.compact
  def __call__(self, x: spec.Tensor, *, train: bool = False) -> spec.Tensor:
    # Patch extraction
    x = nn.Conv(
        self.width,
        self.patch_size,
        strides=self.patch_size,
        padding='VALID',
        name='embedding')(
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
        dropout_rate=dropout_rate,
        name='Transformer')(
            x, train=not train)

    x = jnp.mean(x, axis=1)

    if self.rep_size:
      rep_size = self.width if self.rep_size is True else self.rep_size  # pylint: disable=g-bool-id-comparison
      hid = nn.Dense(rep_size, name='pre_logits')
      x = nn.tanh(hid(x))

    if self.num_classes:
      kw = {'kernel_init': nn.initializers.zeros} if self.head_zeroinit else {}
      head = nn.Dense(self.num_classes, name='head', **kw)
      x = head(x)

    return x
