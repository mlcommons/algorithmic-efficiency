"""U-Net Model.

Adapted from fastMRI:
https://github.com/facebookresearch/fastMRI/blob/main/fastmri/models/unet.py
"""

from functools import partial
from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from algoperf import init_utils
from algoperf.workloads.dropout_modules import CustomDropout2d, SequentialWithDropout



class UNet(nn.Module):
  r"""U-Net model from
    `"U-net: Convolutional networks
    for biomedical image segmentation"
    <hhttps://arxiv.org/pdf/1505.04597.pdf>`_.
    """

  def __init__(self,
               in_chans: int = 1,
               out_chans: int = 1,
               num_channels: int = 32,
               num_pool_layers: int = 4,
               dropout_rate: Optional[float] = 0.0,
               use_tanh: bool = False,
               use_layer_norm: bool = False) -> None:
    super().__init__()

    self.in_chans = in_chans
    self.out_chans = out_chans
    self.num_channels = num_channels
    self.num_pool_layers = num_pool_layers
    if dropout_rate is None:
      self.dropout_rate = 0.0
    else:
      self.dropout_rate = dropout_rate

    self.down_sample_layers = nn.ModuleList([
        ConvBlock(in_chans,
                  num_channels,
                  use_tanh,
                  use_layer_norm)
    ])
    ch = num_channels
    for _ in range(num_pool_layers - 1):
      self.down_sample_layers.append(
          ConvBlock(ch, ch * 2, use_tanh, use_layer_norm))
      ch *= 2
    self.conv = ConvBlock(ch, ch * 2, use_tanh, use_layer_norm)

    self.up_conv = nn.ModuleList()
    self.up_transpose_conv = nn.ModuleList()

    for _ in range(num_pool_layers - 1):
      self.up_transpose_conv.append(
          TransposeConvBlock(ch * 2, ch, use_tanh, use_layer_norm))
      self.up_conv.append(
          ConvBlock(ch * 2, ch, use_tanh, use_layer_norm))
      ch //= 2

    self.up_transpose_conv.append(
        TransposeConvBlock(ch * 2, ch, use_tanh, use_layer_norm))
    self.up_conv.append(
        SequentialWithDropout(
            ConvBlock(ch * 2, ch, use_tanh, use_layer_norm),
            nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
        ))

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init_utils.pytorch_default_init(m)

  def forward(self, x: Tensor, dropout_rate: Optional[float] = None) -> Tensor:
    if dropout_rate is None:
      dropout_rate = self.dropout_rate

    stack = []
    output = x

    # apply down-sampling layers
    for layer in self.down_sample_layers:
      output = layer(output, dropout_rate)
      stack.append(output)
      output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

    output = self.conv(output, dropout_rate)

    # apply up-sampling layers
    for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
      downsample_layer = stack.pop()
      output = transpose_conv(output)

      # reflect pad on the right/bottom if needed to handle
      # odd input dimensions
      padding = [0, 0, 0, 0]
      if output.shape[-1] != downsample_layer.shape[-1]:
        padding[1] = 1  # padding right
      if output.shape[-2] != downsample_layer.shape[-2]:
        padding[3] = 1  # padding bottom
      if torch.sum(torch.tensor(padding)) != 0:
        output = F.pad(output, padding, "reflect")

      output = torch.cat([output, downsample_layer], dim=1)
      output = conv(output, dropout_rate)

    return output


class ConvBlock(nn.Module):
  # A Convolutional Block that consists of two convolution layers each
  # followed by instance normalization, LeakyReLU activation and dropout_rate.

  def __init__(self,
               in_chans: int,
               out_chans: int,
               use_tanh: bool,
               use_layer_norm: bool) -> None:
    super().__init__()
    self._supports_custom_dropout = True

    if use_layer_norm:
      norm_layer = partial(nn.GroupNorm, 1, eps=1e-6)
    else:
      norm_layer = nn.InstanceNorm2d
    if use_tanh:
      activation_fn = nn.Tanh()
    else:
      activation_fn = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    self.conv_layers = SequentialWithDropout(
        nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
        norm_layer(out_chans),
        activation_fn,
        CustomDropout2d(),
        nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
        norm_layer(out_chans),
        activation_fn,
        CustomDropout2d(),
    )

  def forward(self, x: Tensor, dropout_rate: Optional[float] = None) -> Tensor:
    return self.conv_layers(x, dropout_rate)


class TransposeConvBlock(nn.Module):
  # A Transpose Convolutional Block that consists of one convolution transpose
  # layers followed by instance normalization and LeakyReLU activation.

  def __init__(
      self,
      in_chans: int,
      out_chans: int,
      use_tanh: bool,
      use_layer_norm: bool,
  ):
    super().__init__()
    if use_tanh:
      activation_fn = nn.Tanh()
    else:
      activation_fn = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    self.layers = nn.Sequential(
        nn.ConvTranspose2d(
            in_chans, out_chans, kernel_size=2, stride=2, bias=False),
        nn.InstanceNorm2d(out_chans),
        activation_fn,
    )

  def forward(self, x: Tensor) -> Tensor:
    return self.layers(x)
