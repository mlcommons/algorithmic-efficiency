"""Utilities for initializing parameters.

Note: Code adapted from
https://github.com/google/jax/blob/main/jax/_src/nn/initializers.py
"""
import math

from torch import nn


def pytorch_default_init(module: nn.Module) -> None:
  # Perform lecun_normal initialization.
  fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
  std = math.sqrt(1. / fan_in) / .87962566103423978
  nn.init.trunc_normal_(module.weight.data, std=std)
  if module.bias is not None:
    nn.init.constant_(module.bias.data, 0.)
