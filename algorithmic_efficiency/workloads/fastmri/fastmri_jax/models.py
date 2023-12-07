"""Jax / Flax implementation of FastMRI U-Net.

Forked from
https://github.com/google/init2winit/blob/master/init2winit/model_lib/unet.py

Original implementation:
github.com/facebookresearch/fastMRI/blob/main/fastmri/models/unet.py

Training:
github.com/facebookresearch/fastMRI/blob/main/fastmri/pl_modules/unet_module.py

Data:
github.com/facebookresearch/fastMRI/tree/main/fastmri/data
"""
import functools
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


def _instance_norm2d(x, axes, epsilon=1e-5):
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
  mean = jnp.mean(x, axes)
  mean2 = jnp.mean(jnp.square(x), axes)
  # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
  # to floating point round-off errors.
  var = jnp.maximum(0., mean2 - jnp.square(mean))
  stats_shape = list(x.shape)
  for axis in axes:
    stats_shape[axis] = 1
  mean = mean.reshape(stats_shape)
  var = var.reshape(stats_shape)
  y = x - mean
  mul = jnp.sqrt(var + epsilon)
  y /= mul
  return y


class UNet(nn.Module):
  """Jax / Flax implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.

    out_channels: Number of channels in the output to the U-Net model.
    channels: Number of output channels of the first convolution layer.
    num_pool_layers: Number of down-sampling and up-sampling layers.
    dropout_rate: Dropout probability.
  """
  num_channels: int = 32
  num_pool_layers: int = 4
  dropout_rate: Optional[float] = 0.0  # If None, defaults to 0.0.
  use_tanh: bool = False
  use_layer_norm: bool = False

  @nn.compact
  def __call__(self, x, train=True):
    dropout_rate = self.dropout_rate
    if dropout_rate is None:
      dropout_rate = 0.0

    # pylint: disable=invalid-name
    _ConvBlock = functools.partial(
        ConvBlock,
        dropout_rate=dropout_rate,
        use_tanh=self.use_tanh,
        use_layer_norm=self.use_layer_norm)
    _TransposeConvBlock = functools.partial(
        TransposeConvBlock,
        use_tanh=self.use_tanh,
        use_layer_norm=self.use_layer_norm)

    down_sample_layers = [_ConvBlock(self.num_channels)]

    ch = self.num_channels
    for _ in range(self.num_pool_layers - 1):
      down_sample_layers.append(_ConvBlock(ch * 2))
      ch *= 2
    conv = _ConvBlock(ch * 2)

    up_conv = []
    up_transpose_conv = []
    for _ in range(self.num_pool_layers - 1):
      up_transpose_conv.append(_TransposeConvBlock(ch))
      up_conv.append(_ConvBlock(ch))
      ch //= 2

    up_transpose_conv.append(_TransposeConvBlock(ch))
    up_conv.append(_ConvBlock(ch))

    stack = []
    output = jnp.expand_dims(x, axis=-1)

    # apply down-sampling layers
    for layer in down_sample_layers:
      output = layer(output, train)
      stack.append(output)
      output = nn.avg_pool(output, window_shape=(2, 2), strides=(2, 2))

    output = conv(output, train)

    # apply up-sampling layers
    for transpose_conv, conv in zip(up_transpose_conv, up_conv):
      downsample_layer = stack.pop()
      output = transpose_conv(output)

      # reflect pad on the right/botton if needed to handle odd input dimensions
      padding_right = 0
      padding_bottom = 0
      if output.shape[-2] != downsample_layer.shape[-2]:
        padding_right = 1  # padding right
      if output.shape[-3] != downsample_layer.shape[-3]:
        padding_bottom = 1  # padding bottom

      if padding_right or padding_bottom:
        padding = ((0, 0), (0, padding_bottom), (0, padding_right), (0, 0))
        output = jnp.pad(output, padding, mode='reflect')

      output = jnp.concatenate((output, downsample_layer), axis=-1)
      output = conv(output, train)

    out_channels = 1
    output = nn.Conv(out_channels, kernel_size=(1, 1), strides=(1, 1))(output)
    return output.squeeze(-1)


class ConvBlock(nn.Module):
  """A Convolutional Block.
  out_channels: Number of channels in the output.
  dropout_rate: Dropout probability.
  """
  out_channels: int
  dropout_rate: float
  use_tanh: bool
  use_layer_norm: bool

  @nn.compact
  def __call__(self, x, train=True):
    """Forward function.
    Note: Pytorch is NCHW and jax/flax is NHWC.
    Args:
        x: Input 4D tensor of shape `(N, H, W, in_channels)`.
        train: deterministic or not (use init2winit naming).
    Returns:
        jnp.array: Output tensor of shape `(N, H, W, out_channels)`.
    """
    x = nn.Conv(
        features=self.out_channels,
        kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=False)(
            x)
    if self.use_layer_norm:
      x = nn.LayerNorm(reduction_axes=(1, 2, 3))(x)
    else:
      # DO NOT SUBMIT check that this comment edit is correct
      # InstanceNorm2d was run with no learnable params in reference code
      # so this is a simple normalization along spatial dims.
      x = _instance_norm2d(x, (1, 2))
    if self.use_tanh:
      activation_fn = nn.tanh
    else:
      activation_fn = functools.partial(jax.nn.leaky_relu, negative_slope=0.2)
    x = activation_fn(x)
    # Ref code uses dropout2d which applies the same mask for the entire channel
    # Replicated by using broadcast dims to have the same filter on HW
    x = nn.Dropout(
        self.dropout_rate, broadcast_dims=(1, 2), deterministic=not train)(
            x)
    x = nn.Conv(
        features=self.out_channels,
        kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=False)(
            x)
    if self.use_layer_norm:
      x = nn.LayerNorm(reduction_axes=(1, 2, 3))(x)
    else:
      x = _instance_norm2d(x, (1, 2))
    x = activation_fn(x)
    x = nn.Dropout(
        self.dropout_rate, broadcast_dims=(1, 2), deterministic=not train)(
            x)
    return x


class TransposeConvBlock(nn.Module):
  """A Transpose Convolutional Block.
  out_channels: Number of channels in the output.
  """
  out_channels: int
  use_tanh: bool
  use_layer_norm: bool

  @nn.compact
  def __call__(self, x):
    """Forward function.
    Args:
        x: Input 4D tensor of shape `(N, H, W, in_channels)`.
    Returns:
        jnp.array: Output tensor of shape `(N, H*2, W*2, out_channels)`.
    """
    x = nn.ConvTranspose(
        self.out_channels, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(
            x)
    x = _instance_norm2d(x, (1, 2))
    if self.use_tanh:
      activation_fn = nn.tanh
    else:
      activation_fn = functools.partial(jax.nn.leaky_relu, negative_slope=0.2)
    x = activation_fn(x)
    return x
