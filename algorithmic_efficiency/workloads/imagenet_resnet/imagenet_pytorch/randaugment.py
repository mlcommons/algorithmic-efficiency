"""Code for RandAugmentation.

Adapted from:
https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import PIL
import torch
from torch import Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


def cutout(img, pad_size, fill):
  image_width, image_height = img.size
  cutout_center_width = np.random.uniform(image_width)
  cutout_center_height = np.random.uniform(image_height)

  lower_pad = max(0, cutout_center_height - pad_size)
  upper_pad = max(0, image_height - cutout_center_height - pad_size)
  left_pad = max(0, cutout_center_width - pad_size)
  right_pad = max(0, image_width - cutout_center_width - pad_size)

  x0 = right_pad
  y0 = upper_pad
  x1 = left_pad
  y1 = lower_pad

  xy = (x0, y0, x1, y1)
  img = img.copy()
  PIL.ImageDraw.Draw(img).rectangle(xy, fill)
  return img


def solarize(img: Tensor, threshold: float) -> Tensor:
  img = np.array(img)
  new_img = np.where(img < threshold, img, 255. - img)
  return PIL.Image.fromarray(new_img.astype(np.uint8))


def solarize_add(img: Tensor, addition: int = 0) -> Tensor:
  threshold = 128
  img = np.array(img)
  added_img = img.astype(np.int64) + addition
  added_img = np.clip(added_img, 0, 255).astype(np.uint8)
  new_img = np.where(img < threshold, added_img, img)
  return PIL.Image.fromarray(new_img)


def _apply_op(
    img: Tensor,
    op_name: str,
    magnitude: float,
    interpolation: InterpolationMode,
    fill: Optional[List[float]],
):
  if op_name == "ShearX":
    # magnitude should be arctan(magnitude)
    # official autoaug: (1, level, 0, 0, 1, 0)
    # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
    # compared to
    # torchvision:      (1, tan(level), 0, 0, 1, 0)
    # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
    img = F.affine(
        img,
        angle=0.0,
        translate=[0, 0],
        scale=1.0,
        shear=[math.degrees(math.atan(magnitude)), 0.0],
        interpolation=interpolation,
        fill=fill,
        center=[0, 0],
    )
  elif op_name == "ShearY":
    # magnitude should be arctan(magnitude)
    # See above
    img = F.affine(
        img,
        angle=0.0,
        translate=[0, 0],
        scale=1.0,
        shear=[0.0, math.degrees(math.atan(magnitude))],
        interpolation=interpolation,
        fill=fill,
        center=[0, 0],
    )
  elif op_name == "TranslateX":
    img = F.affine(
        img,
        angle=0.0,
        translate=[int(magnitude), 0],
        scale=1.0,
        interpolation=interpolation,
        shear=[0.0, 0.0],
        fill=fill,
    )
  elif op_name == "TranslateY":
    img = F.affine(
        img,
        angle=0.0,
        translate=[0, int(magnitude)],
        scale=1.0,
        interpolation=interpolation,
        shear=[0.0, 0.0],
        fill=fill,
    )
  elif op_name == "Rotate":
    img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
  elif op_name == "Brightness":
    img = F.adjust_brightness(img, magnitude)
  elif op_name == "Color":
    img = F.adjust_saturation(img, magnitude)
  elif op_name == "Contrast":
    img = F.adjust_contrast(img, magnitude)
  elif op_name == "Sharpness":
    img = F.adjust_sharpness(img, magnitude)
  elif op_name == "Posterize":
    img = F.posterize(img, int(magnitude))
  elif op_name == "Cutout":
    img = cutout(img, magnitude, fill=fill)
  elif op_name == "SolarizeAdd":
    img = solarize_add(img, int(magnitude))
  elif op_name == "Solarize":
    img = solarize(img, magnitude)
  elif op_name == "AutoContrast":
    img = F.autocontrast(img)
  elif op_name == "Equalize":
    img = F.equalize(img)
  elif op_name == "Invert":
    img = F.invert(img)
  elif op_name == "Identity":
    pass
  else:
    raise ValueError(f"The provided operator {op_name} is not recognized.")
  return img


class RandAugment(torch.nn.Module):

  def __init__(
      self,
      num_ops: int = 2,
      interpolation: InterpolationMode = InterpolationMode.NEAREST,
      fill: Optional[List[float]] = None,
  ) -> None:
    super().__init__()
    self.num_ops = num_ops
    self.interpolation = interpolation
    self.fill = fill

  def _augmentation_space(self) -> Dict[str, Tuple[Tensor, bool]]:
    return {
        # op_name: (magnitudes, signed)
        "ShearX": (torch.tensor(0.3), True),
        "ShearY": (torch.tensor(0.3), True),
        "TranslateX": (torch.tensor(100), True),
        "TranslateY": (torch.tensor(100), True),
        "Rotate": (torch.tensor(30), True),
        "Brightness": (torch.tensor(1.9), False),
        "Color": (torch.tensor(1.9), False),
        "Contrast": (torch.tensor(1.9), False),
        "Sharpness": (torch.tensor(1.9), False),
        "Posterize": (torch.tensor(4), False),
        "Solarize": (torch.tensor(256), False),
        "SolarizeAdd": (torch.tensor(110), False),
        "AutoContrast": (torch.tensor(0.0), False),
        "Equalize": (torch.tensor(0.0), False),
        "Cutout": (torch.tensor(40.0), False),
    }

  def forward(self, img: Tensor) -> Tensor:
    fill = self.fill if self.fill is not None else 128
    channels, _, _ = F.get_dimensions(img)
    if isinstance(img, Tensor):
      if isinstance(fill, (int, float)):
        fill = [float(fill)] * channels
      elif fill is not None:
        fill = [float(f) for f in fill]

    op_meta = self._augmentation_space()
    for _ in range(self.num_ops):
      op_index = int(torch.randint(len(op_meta), (1,)).item())
      op_name = list(op_meta.keys())[op_index]
      magnitude, signed = op_meta[op_name]
      magnitude = float(magnitude)
      if signed and torch.randint(2, (1,)):
        magnitude *= -1.0
      img = _apply_op(
          img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

    return img

  def __repr__(self) -> str:
    s = (f"{self.__class__.__name__}("
         f"num_ops={self.num_ops}"
         f", magnitude={self.magnitude}"
         f", num_magnitude_bins={self.num_magnitude_bins}"
         f", interpolation={self.interpolation}"
         f", fill={self.fill}"
         f")")
    return s
