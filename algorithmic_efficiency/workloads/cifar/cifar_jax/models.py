"""Jax implementation of ResNet V1 for CIFAR.

Adapted from Flax example:
https://github.com/google/flax/blob/main/examples/imagenet/models.py.
"""

import functools
from typing import Any, Callable, Tuple

from flax import linen as nn
import jax.numpy as jnp

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax.models import \
    ResNetBlock

ModuleDef = nn.Module


class ResNet(nn.Module):
  stage_sizes: Tuple[int]
  block_cls: ModuleDef
  num_classes: int = 10
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu

  @nn.compact
  def __call__(self,
               x: spec.Tensor,
               update_batch_norm: bool = True,
               use_running_average_bn: bool = None) -> spec.Tensor:
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)

    # Preserve default behavior for backwards compatibility
    if use_running_average_bn is None:
      use_running_average_bn = not update_batch_norm
    norm = functools.partial(
        nn.BatchNorm,
        use_running_average=use_running_average_bn,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype)

    x = conv(
        self.num_filters, (3, 3), (1, 1),
        padding=[(1, 1), (1, 1)],
        name='Conv_init')(
            x)
    x = norm(name='BatchNorm_init')(x)
    x = nn.relu(x)
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
    x = nn.avg_pool(x, (4, 4), strides=(4, 4))
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.normal(),
        dtype=self.dtype)(
            x)
    return x


ResNet18 = functools.partial(
    ResNet, stage_sizes=(2, 2, 2, 2), block_cls=ResNetBlock)
