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

import flax.linen as nn
import jax
import jax.numpy as jnp


def _compute_stats(x, axes):
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
  mean = jnp.mean(x, axes)
  mean2 = jnp.mean(jnp.square(x), axes)
  # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
  # to floating point round-off errors.
  var = jnp.maximum(0., mean2 - jnp.square(mean))
  return mean, var


def _normalize(x, axes, mean, var, epsilon):
  stats_shape = list(x.shape)
  for axis in axes:
    stats_shape[axis] = 1
  mean = mean.reshape(stats_shape)
  var = var.reshape(stats_shape)
  y = x - mean
  mul = jnp.sqrt(var + epsilon)
  y /= mul
  return y


def _simple_instance_norm2d(x, axes, epsilon=1e-5):
  mean, var = _compute_stats(x, axes)
  return _normalize(x, axes, mean, var, epsilon)


class UNet(nn.Module):
  """Jax / Flax implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.

    out_channels: Number of channels in the output to the U-Net model.
    channels: Number of output channels of the first convolution layer.
    num_pool_layers: Number of down-sampling and up-sampling layers.
    drop_prob: Dropout probability.
  """
  out_channels: int = 1
  channels: int = 32
  num_pool_layers: int = 4
  drop_prob: float = 0.0

  @nn.compact
  def __call__(self, x, train=True):
    down_sample_layers = [ConvBlock(self.channels, self.drop_prob)]

    ch = self.channels
    for _ in range(self.num_pool_layers - 1):
      down_sample_layers.append(ConvBlock(ch * 2, self.drop_prob))
      ch *= 2
    conv = ConvBlock(ch * 2, self.drop_prob)

    up_conv = []
    up_transpose_conv = []
    for _ in range(self.num_pool_layers - 1):
      up_transpose_conv.append(TransposeConvBlock(ch))
      up_conv.append(ConvBlock(ch, self.drop_prob))
      ch //= 2

    up_transpose_conv.append(TransposeConvBlock(ch))
    up_conv.append(ConvBlock(ch, self.drop_prob))

    final_conv = nn.Conv(self.out_channels, kernel_size=(1, 1), strides=(1, 1))

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

    output = final_conv(output)

    return output.squeeze(-1)


class ConvBlock(nn.Module):
  """A Convolutional Block.
  out_channels: Number of channels in the output.
  drop_prob: Dropout probability.
  """
  out_channels: int
  drop_prob: float

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
    # InstanceNorm2d was run with no learnable params in reference code
    # so this is a simple normalization along channels
    x = _simple_instance_norm2d(x, (1, 2))
    x = jax.nn.leaky_relu(x, negative_slope=0.2)
    # Ref code uses dropout2d which applies the same mask for the entire channel
    # Replicated by using broadcast dims to have the same filter on HW
    x = nn.Dropout(
        self.drop_prob, broadcast_dims=(1, 2), deterministic=not train)(
            x)
    x = nn.Conv(
        features=self.out_channels,
        kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=False)(
            x)
    x = _simple_instance_norm2d(x, (1, 2))
    x = jax.nn.leaky_relu(x, negative_slope=0.2)
    x = nn.Dropout(
        self.drop_prob, broadcast_dims=(1, 2), deterministic=not train)(
            x)

    return x


class TransposeConvBlock(nn.Module):
  """A Transpose Convolutional Block.
  out_channels: Number of channels in the output.
  """
  out_channels: int

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
    x = _simple_instance_norm2d(x, (1, 2))
    x = jax.nn.leaky_relu(x, negative_slope=0.2)

    return x
