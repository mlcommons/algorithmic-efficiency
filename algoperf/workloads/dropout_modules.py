"""Custom classes to support a dynamic modulized dropout, see issue??TODO"""

from torch import Tensor
from torch import nn
import torch.nn.functional as F


class CustomDropout(nn.Module):
  """A module around torch.nn.functional.dropout."""
  def __init__(self):
    super().__init__()
    self._supports_custom_dropout = True

  def forward(self, input: Tensor, p: float) -> Tensor:
    return F.dropout(input, p, training=self.training)


class CustomDropout2d(nn.Module):
  """A module around torch.nn.functional.dropout2d."""
  def __init__(self):
    super().__init__()
    self._supports_custom_dropout = True

  def forward(self, input: Tensor, p: float) -> Tensor:
    return F.dropout2d(input, p, training=self.training)


class SequentialWithDropout(nn.Sequential):
  """Sequential of modules with dropout."""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._supports_custom_dropout = True

  def forward(self, x, p):
    for module in self:
      # if isinstance(module, (CustomDropout, SequentialWithDropout, DenseBlockWithDropout)):
      if getattr(module, '_supports_custom_dropout', False):  # TODO (nico): improve
        x = module(x, p)
      else:
        x = module(x) 
    return x
