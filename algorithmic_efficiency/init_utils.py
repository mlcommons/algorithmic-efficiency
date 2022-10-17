from torch import nn
import math
from functools import partial
from flax.nn.initializers import variance_scaling

jax_kaiming_non_trunc_normal = partial(variance_scaling, 2.0, "fan_in", "normal")
jax_kaiming_custom_uniform = partial(variance_scaling, 2.0, "fan_in", "normal")


def pytorch_default_init(module: nn.Module) -> None:
  fan_in = module.in_features
  std = math.sqrt(1. / fan_in) / .87962566103423978
  nn.init.trunc_normal_(module.weight.data, std=std)
  if module.bias is not None:
    nn.init.constant_(module.bias.data, 0.)
