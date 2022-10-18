import math

from torch import nn


def pytorch_default_init(module: nn.Module) -> None:
  fan_in = module.in_features
  std = math.sqrt(1. / fan_in) / .87962566103423978
  nn.init.trunc_normal_(module.weight.data, std=std)
  if module.bias is not None:
    nn.init.constant_(module.bias.data, 0.)
