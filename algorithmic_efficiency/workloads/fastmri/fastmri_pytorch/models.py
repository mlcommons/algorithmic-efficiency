"""U-Net Model.

Adapted from fastMRI:
https://github.com/facebookresearch/fastMRI/blob/main/fastmri/models/unet.py
"""

from typing import Any

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from algorithmic_efficiency import init_utils


class UNet(nn.Module):
  r"""U-Net model from
    `"U-net: Convolutional networks
    for biomedical image segmentation"
    <hhttps://arxiv.org/pdf/1505.04597.pdf>`_.
    """

  def __init__(self,
               in_chans: int = 1,
               out_chans: int = 1,
               chans: int = 32,
               num_pool_layers: int = 4,
               dropout: float = 0.0) -> None:
    super().__init__()

    self.in_chans = in_chans
    self.out_chans = out_chans
    self.chans = chans
    self.num_pool_layers = num_pool_layers
    self.dropout = dropout

    self.down_sample_layers = nn.ModuleList(
        [ConvBlock(in_chans, chans, dropout)])
    ch = chans
    for _ in range(num_pool_layers - 1):
      self.down_sample_layers.append(ConvBlock(ch, ch * 2, dropout))
      ch *= 2
    self.conv = ConvBlock(ch, ch * 2, dropout)

    self.up_conv = nn.ModuleList()
    self.up_transpose_conv = nn.ModuleList()
    for _ in range(num_pool_layers - 1):
      self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
      self.up_conv.append(ConvBlock(ch * 2, ch, dropout))
      ch //= 2

    self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
    self.up_conv.append(
        nn.Sequential(
            ConvBlock(ch * 2, ch, dropout),
            nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
        ))

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init_utils.pytorch_default_init(m)

  def forward(self, x: Tensor) -> Tensor:
    stack = []
    output = x

    # apply down-sampling layers
    for layer in self.down_sample_layers:
      output = layer(output)
      stack.append(output)
      output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

    output = self.conv(output)

    # apply up-sampling layers
    for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
      downsample_layer = stack.pop()
      output = transpose_conv(output)

      # reflect pad on the right/botton if needed to handle
      # odd input dimensions
      padding = [0, 0, 0, 0]
      if output.shape[-1] != downsample_layer.shape[-1]:
        padding[1] = 1  # padding right
      if output.shape[-2] != downsample_layer.shape[-2]:
        padding[3] = 1  # padding bottom
      if torch.sum(torch.tensor(padding)) != 0:
        output = F.pad(output, padding, "reflect")

      output = torch.cat([output, downsample_layer], dim=1)
      output = conv(output)

    return output


class ConvBlock(nn.Module):
  # A Convolutional Block that consists of two convolution layers each
  # followed by instance normalization, LeakyReLU activation and dropout.

  def __init__(self,
               in_chans: int,
               out_chans: int,
               dropout: float = 0.0) -> None:
    super().__init__()

    self.in_chans = in_chans
    self.out_chans = out_chans
    self.dropout = dropout

    self.conv_layers = nn.Sequential(
        nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(out_chans),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Dropout2d(dropout),
        nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(out_chans),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Dropout2d(dropout),
    )

  def forward(self, x: Tensor) -> Tensor:
    return self.conv_layers(x)


class TransposeConvBlock(nn.Module):
  # A Transpose Convolutional Block that consists of one convolution transpose
  # layers followed by instance normalization and LeakyReLU activation.

  def __init__(self, in_chans: int, out_chans: int):
    super().__init__()

    self.in_chans = in_chans
    self.out_chans = out_chans

    self.layers = nn.Sequential(
        nn.ConvTranspose2d(
            in_chans, out_chans, kernel_size=2, stride=2, bias=False),
        nn.InstanceNorm2d(out_chans),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )

  def forward(self, x: Tensor) -> Tensor:
    return self.layers(x)
