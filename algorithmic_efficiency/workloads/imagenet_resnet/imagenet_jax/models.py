"""Flax implementation of ResNet V1.

Forked from Flax example which can be found here:
https://github.com/google/flax/blob/main/examples/imagenet/models.py
"""

from functools import partial
from typing import Any, Callable, Tuple

from flax import linen as nn
import jax.numpy as jnp

ModuleDef = nn.Module


class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(
          self.filters, (1, 1), self.strides, name='conv_proj')(
              residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(
          self.filters * 4, (1, 1), self.strides, name='conv_proj')(
              residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1."""
  stage_sizes: Tuple[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu

  @nn.compact
  def __call__(self, x, update_batch_norm: bool = True):
    conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = partial(
        nn.BatchNorm,
        use_running_average=not update_batch_norm,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype)

    x = conv(
        self.num_filters, (7, 7), (2, 2),
        padding=[(3, 3), (3, 3)],
        name='conv_init')(
            x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(
            self.num_filters * 2**i,
            strides=strides,
            conv=conv,
            norm=norm,
            act=self.act)(
                x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    x = nn.log_softmax(x)
    return x


ResNet18 = partial(ResNet, stage_sizes=(2, 2, 2, 2), block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=(3, 4, 6, 3), block_cls=ResNetBlock)
ResNet50 = partial(
    ResNet, stage_sizes=(3, 4, 6, 3), block_cls=BottleneckResNetBlock)
ResNet101 = partial(
    ResNet, stage_sizes=(3, 4, 23, 3), block_cls=BottleneckResNetBlock)
ResNet152 = partial(
    ResNet, stage_sizes=(3, 8, 36, 3), block_cls=BottleneckResNetBlock)
ResNet200 = partial(
    ResNet, stage_sizes=(3, 24, 36, 3), block_cls=BottleneckResNetBlock)

# Used for testing only.
_ResNet1 = partial(ResNet, stage_sizes=(1,), block_cls=ResNetBlock)
