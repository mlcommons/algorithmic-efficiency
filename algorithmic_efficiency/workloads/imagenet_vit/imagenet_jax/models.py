"""A refactored and simplified ViT.

NOTE: Forked from
https://github.com/google/init2winit/blob/master/init2winit/model_lib/vit.py,
originally from https://github.com/google/big_vision with modifications noted.
"""

from typing import Optional, Sequence, Union

from flax import linen as nn
import jax.numpy as jnp
import numpy as np


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


def get_posemb(self, emb_type, seqshape, width, name, dtype=jnp.float32):
  if emb_type == 'learn':
    return self.param(name,
                      nn.initializers.normal(stddev=1 / np.sqrt(width)),
                      (1, np.prod(seqshape), width),
                      dtype)
  elif emb_type == 'sincos2d':
    return posemb_sincos_2d(*seqshape, width, dtype=dtype)
  else:
    raise ValueError(f'Unknown posemb type: {emb_type}')


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, train=True):
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )

    d = x.shape[2]
    x = nn.Dense(self.mlp_dim or 4 * d, **inits)(x)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout)(x, train)
    x = nn.Dense(d, **inits)(x)
    return x


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, train=True):
    out = {}
    y = nn.LayerNorm(name='LayerNorm_0')(x)
    y = out['sa'] = nn.SelfAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=train,
        name='MultiHeadDotProductAttention_1',
    )(
        y)
    y = nn.Dropout(rate=self.dropout)(y, train)
    x = out['+sa'] = x + y

    y = nn.LayerNorm(name='LayerNorm_2')(x)
    y = out['mlp'] = MlpBlock(
        mlp_dim=self.mlp_dim,
        dropout=self.dropout,
        name='MlpBlock_3',
    )(y, train)
    y = nn.Dropout(rate=self.dropout)(y, train)
    x = out['+mlp'] = x + y
    return x, out


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  depth: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, train=True):
    out = {}

    # Input Encoder
    for lyr in range(self.depth):
      block = Encoder1DBlock(
          name=f'encoderblock_{lyr}',
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout=self.dropout)
      x, out[f'block{lyr:02d}'] = block(x, train)
    out['pre_ln'] = x  # Alias for last block, but without the number in it.

    return nn.LayerNorm(name='encoder_norm')(x), out


class MAPHead(nn.Module):
  """Multihead Attention Pooling."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12

  @nn.compact
  def __call__(self, x):
    # TODO(lbeyer): condition on GAP(x)
    n, _, d = x.shape
    probe = self.param('probe',
                       nn.initializers.xavier_uniform(), (1, 1, d),
                       x.dtype)
    probe = jnp.tile(probe, [n, 1, 1])

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform())(probe, x)

    # TODO(lbeyer): dropout on head?
    y = nn.LayerNorm()(x)
    x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
    return x[:, 0]


class ViT(nn.Module):
  """ViT model."""

  num_classes: int
  patch_size: Sequence[int] = (16, 16)
  width: int = 768
  depth: int = 12
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  posemb: str = 'sincos2d'  # Can also be "learn"
  rep_size: Union[int, bool] = True
  dropout: float = 0.0
  pool_type: str = 'gap'  # Can also be 'map' or 'tok'
  reinit: Optional[Sequence[str]] = None
  head_zeroinit: bool = True

  @nn.compact
  def __call__(self, x, *, train=False):
    out = {}

    # Patch extraction
    x = out['stem'] = nn.Conv(
        self.width,
        self.patch_size,
        strides=self.patch_size,
        padding='VALID',
        name='embedding')(
            x)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # Add posemb before adding extra token.
    x = out['with_posemb'] = x + get_posemb(
        self, self.posemb, (h, w), c, 'pos_embedding', x.dtype)

    if self.pool_type == 'tok':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

    n, l, c = x.shape  # pylint: disable=unused-variable
    x = nn.Dropout(rate=self.dropout)(x, not train)

    x, out['encoder'] = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout,
        name='Transformer')(
            x, train=not train)
    encoded = out['encoded'] = x

    if self.pool_type == 'map':
      x = out['head_input'] = MAPHead(
          num_heads=self.num_heads, mlp_dim=self.mlp_dim)(
              x)
    elif self.pool_type == 'gap':
      x = out['head_input'] = jnp.mean(x, axis=1)
    elif self.pool_type == '0':
      x = out['head_input'] = x[:, 0]
    elif self.pool_type == 'tok':
      x = out['head_input'] = x[:, 0]
      encoded = encoded[:, 1:]
    else:
      raise ValueError(f'Unknown pool type: "{self.pool_type}"')

    x_2d = jnp.reshape(encoded, [n, h, w, -1])

    if self.rep_size:
      rep_size = self.width if self.rep_size is True else self.rep_size  # pylint: disable=g-bool-id-comparison
      hid = nn.Dense(rep_size, name='pre_logits')
      # NOTE: In the past we did not include tanh in pre_logits.
      # For few-shot, it should not matter much, as it whitens anyways.
      x_2d = nn.tanh(hid(x_2d))
      x = nn.tanh(hid(x))

    out['pre_logits_2d'] = x_2d
    out['pre_logits'] = x

    if self.num_classes:
      kw = {'kernel_init': nn.initializers.zeros} if self.head_zeroinit else {}
      head = nn.Dense(self.num_classes, name='head', **kw)
      x_2d = out['logits_2d'] = head(x_2d)
      x = out['logits'] = head(x)

    return x
