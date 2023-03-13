"""PyTorch implementation of ResNet for CIFAR.

Adapted from torchvision:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.
"""

import collections
from typing import Any, Callable, List, Optional, Type, Union

import torch
from torch import nn

from algorithmic_efficiency import spec
from algorithmic_efficiency.init_utils import pytorch_default_init
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_pytorch.models import \
    BasicBlock
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_pytorch.models import \
    Bottleneck
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_pytorch.models import \
    conv1x1


class ResNet(nn.Module):

  def __init__(self,
               block: Type[Union[BasicBlock, Bottleneck]],
               layers: List[int],
               num_classes: int = 10,
               groups: int = 1,
               width_per_group: int = 64,
               replace_stride_with_dilation: Optional[List[bool]] = None,
               norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
    super().__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    self._norm_layer = norm_layer

    self.inplanes = 64
    self.dilation = 1
    if replace_stride_with_dilation is None:
      # Each element in the tuple indicates if we should replace
      # the 2x2 stride with a dilated convolution instead.
      replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
      raise ValueError(
          'replace_stride_with_dilation should be None '
          f'or a 3-element tuple, got {replace_stride_with_dilation}')
    self.groups = groups
    self.base_width = width_per_group
    self.conv1 = nn.Conv2d(
        3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = norm_layer(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(
        block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
    self.layer3 = self._make_layer(
        block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
    self.layer4 = self._make_layer(
        block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
    self.fc = nn.Linear(512 * block.expansion, num_classes)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        pytorch_default_init(m)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    nn.init.normal_(self.fc.weight, std=1e-2)
    nn.init.constant_(self.fc.bias, 0.)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros,
    # and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to
    # https://arxiv.org/abs/1706.02677.
    for m in self.modules():
      if isinstance(m, Bottleneck):
        nn.init.constant_(m.bn3.weight, 0)
      elif isinstance(m, BasicBlock):
        nn.init.constant_(m.bn2.weight, 0)

  def _make_layer(self,
                  block: Type[Union[BasicBlock, Bottleneck]],
                  planes: int,
                  blocks: int,
                  stride: int = 1,
                  dilate: bool = False) -> nn.Sequential:
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = torch.nn.Sequential(
          collections.OrderedDict([
              ("conv", conv1x1(self.inplanes, planes * block.expansion,
                               stride)),
              ("bn", norm_layer(planes * block.expansion)),
          ]))

    layers = []
    layers.append(
        block(self.inplanes,
              planes,
              stride,
              downsample,
              self.groups,
              self.base_width,
              previous_dilation,
              norm_layer))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
          block(
              self.inplanes,
              planes,
              groups=self.groups,
              base_width=self.base_width,
              dilation=self.dilation,
              norm_layer=norm_layer))

    return nn.Sequential(*layers)

  def forward(self, x: spec.Tensor) -> spec.Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = torch.nn.functional.avg_pool2d(x, 4)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x


def resnet18(**kwargs: Any) -> ResNet:
  return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
