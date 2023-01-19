"""Utilities for Jax and Pytorch transfer operations."""

import jax.dlpack
import torch

from algorithmic_efficiency import spec


def jax_to_pytorch(x: spec.Tensor, take_ownership: bool = False) -> spec.Tensor:
  return torch.utils.dlpack.from_dlpack(
      jax.dlpack.to_dlpack(x, take_ownership=take_ownership))


def pytorch_to_jax(x: torch.Tensor) -> spec.Tensor:
  x = x.contiguous()  # https://github.com/google/jax/issues/8082
  return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))
