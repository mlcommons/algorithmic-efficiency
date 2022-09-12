"""Discrete Fourier transforms and related functions.

Adapted from FastMRI:
https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
"""

from typing import List, Optional

import torch
from torch import Tensor
import torch.fft


def ifft2c_new(data: Tensor, norm: str = 'ortho') -> Tensor:
  if not data.shape[-1] == 2:
    raise ValueError('Tensor does not have separate complex dim.')

  data = ifftshift(data, dim=[-3, -2])
  data = torch.view_as_real(
      torch.fft.ifftn(  # type: ignore
          torch.view_as_complex(data), dim=(-2, -1), norm=norm))
  data = fftshift(data, dim=[-3, -2])

  return data


def roll_one_dim(x: Tensor, shift: int, dim: int) -> Tensor:
  shift = shift % x.size(dim)
  if shift == 0:
    return x

  left = x.narrow(dim, 0, x.size(dim) - shift)
  right = x.narrow(dim, x.size(dim) - shift, shift)

  return torch.cat((right, left), dim=dim)


def roll(x: Tensor, shift: List[int], dim: List[int]) -> Tensor:
  if len(shift) != len(dim):
    raise ValueError('len(shift) must match len(dim)')

  for (s, d) in zip(shift, dim):
    x = roll_one_dim(x, s, d)

  return x


def fftshift(x: Tensor, dim: Optional[List[int]] = None) -> Tensor:
  if dim is None:
    # this weird code is necessary for torch.jit.script typing
    dim = [0] * (x.dim())
    for i in range(1, x.dim()):
      dim[i] = i

  # also necessary for torch.jit.script
  shift = [0] * len(dim)
  for i, dim_num in enumerate(dim):
    shift[i] = x.shape[dim_num] // 2

  return roll(x, shift, dim)


def ifftshift(x: Tensor, dim: Optional[List[int]] = None) -> Tensor:
  if dim is None:
    # this weird code is necessary for torch.jit.script typing
    dim = [0] * (x.dim())
    for i in range(1, x.dim()):
      dim[i] = i

  # also necessary for torch.jit.script
  shift = [0] * len(dim)
  for i, dim_num in enumerate(dim):
    shift[i] = (x.shape[dim_num] + 1) // 2

  return roll(x, shift, dim)
