"""Jax implementation of ResNet V1.

Adapted from Flax example:
https://github.com/google/flax/blob/main/examples/imagenet/models.py
"""

import functools
from typing import Any, Callable, Tuple

from flax import linen as nn
import jax.numpy as jnp

from algorithmic_efficiency import spec

ModuleDef = nn.Module


class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x: spec.Tensor) -> spec.Tensor:
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape or self.strides != (1, 1):
      residual = self.conv(
          self.filters, (1, 1), self.strides, name='Conv_proj')(
              residual)
      residual = self.norm(name='BatchNorm_proj')(residual)

    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x: spec.Tensor) -> spec.Tensor:
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape or self.strides != (1, 1):
      residual = self.conv(
          self.filters * 4, (1, 1), self.strides, name='Conv_proj')(
              residual)
      residual = self.norm(name='BatchNorm_proj')(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  stage_sizes: Tuple[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu

  @nn.compact
  def __call__(self,
               x: spec.Tensor,
               update_batch_norm: bool = True) -> spec.Tensor:
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not update_batch_norm,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype)

    x = conv(
        self.num_filters, (7, 7), (2, 2),
        padding=[(3, 3), (3, 3)],
        name='Conv_init')(
            x)
    x = norm(name='BatchNorm_init')(x)
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
    x = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.normal(),
        dtype=self.dtype)(
            x)
    return x


ResNet18 = functools.partial(
    ResNet, stage_sizes=(2, 2, 2, 2), block_cls=ResNetBlock)
ResNet50 = functools.partial(
    ResNet, stage_sizes=(3, 4, 6, 3), block_cls=BottleneckResNetBlock)
