import os
from typing import Tuple

import jax
import torch

from algorithmic_efficiency import spec


def pytorch_setup() -> Tuple[bool, int, torch.device, int]:
    use_pytorch_ddp = "LOCAL_RANK" in os.environ
    rank = int(os.environ["LOCAL_RANK"]) if use_pytorch_ddp else 0
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    return use_pytorch_ddp, rank, device, n_gpus


def jax_to_pytorch(x: spec.Tensor, take_ownership: bool = False):
    return torch.utils.dlpack.from_dlpack(
        jax.dlpack.to_dlpack(x, take_ownership=take_ownership)
    )


def pytorch_to_jax(x: torch.Tensor):
    x = x.contiguous()  # https://github.com/google/jax/issues/8082
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))
