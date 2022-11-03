import torch
import numpy as np
from ffcv.pipeline.operation import Operation
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State


class ToInputTensor(Operation):
  """Convert from Numpy array to PyTorch Tensor."""

  def __init__(self):
    super().__init__()

  def generate_code(self) -> Callable:
    def to_tensor(inp, dst):
      nump_array = np.asarray(inp, dtype=np.uint8)
      if nump_array.ndim < 3:
        nump_array = np.expand_dims(nump_array, axis=-1)
      nump_array = np.rollaxis(nump_array, 2)
      return torch.from_numpy(nump_array.copy())
    return to_tensor

  def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
    new_dtype = torch.from_numpy(np.empty((), dtype=previous_state.dtype)).dtype
    return replace(previous_state, jit_mode=False, dtype=new_dtype), None


class ToLabelTensor(Operation):
  """Convert from Numpy array to PyTorch Tensor."""

  def __init__(self):
    super().__init__()

  def generate_code(self) -> Callable:
    def to_tensor(inp, dst):
      return torch.from_numpy(inp)

    return to_tensor

  def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
    new_dtype = torch.from_numpy(np.empty((), dtype=previous_state.dtype)).dtype
    return replace(previous_state, jit_mode=False, dtype=new_dtype), None


class ToDevice(Operation):

  def __init__(self, device, non_blocking=True):
    super().__init__()
    self.device = device
    self.non_blocking = non_blocking

  def generate_code(self) -> Callable:
    def to_device(inp, dst):
      dst = dst[:inp.shape[0]]
      dst.copy_(inp, non_blocking=self.non_blocking)
      return dst
    return to_device

  def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
    return replace(previous_state, device=self.device), AllocationQuery(previous_state.shape,
                                                                        dtype=previous_state.dtype, device=self.device)


class ToTorchImage(Operation):

  def __init__(self):
    super().__init__()
    self.enable_int16conv = False

  def generate_code(self) -> Callable:
    do_conv = self.enable_int16conv

    def to_torch_image(inp: torch.Tensor, dst):
      # Returns a permuted view of the same tensor
      inp = inp.permute([0, 3, 1, 2])
      dst[:inp.shape[0]] = inp.contiguous()
      return dst[:inp.shape[0]]

    return to_torch_image

  def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
    H, W, C = previous_state.shape
    new_type = previous_state.dtype
    alloc = AllocationQuery((C, H, W), dtype=new_type)
    return replace(previous_state, shape=(C, H, W), dtype=new_type), alloc


def fast_collate(batch, memory_format=torch.contiguous_format):
  imgs = [img[0] for img in batch]
  targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
  w = imgs[0].size[0]
  h = imgs[0].size[1]
  tensor = torch.zeros(
      (len(imgs), 3, h, w),
      dtype=torch.uint8).contiguous(memory_format=memory_format)
  for i, img in enumerate(imgs):
    nump_array = np.asarray(img, dtype=np.uint8)
    if nump_array.ndim < 3:
      nump_array = np.expand_dims(nump_array, axis=-1)
    nump_array = np.rollaxis(nump_array, 2)
    tensor[i] += torch.from_numpy(nump_array.copy())
  return tensor, targets
