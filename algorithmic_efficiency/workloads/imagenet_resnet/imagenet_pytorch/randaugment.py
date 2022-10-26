"""PyTorch implementation of RandAugmentation.

Adapted from:
https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import PIL
import torch
from torch import Tensor
from algorithmic_efficiency import spec
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


def cutout(img: spec.Tensor, pad_size: int, fill: float) -> spec.Tensor:
  image_width, image_height = img.size
  x0 = np.random.uniform(image_width)
  y0 = np.random.uniform(image_height)

  pad_size = pad_size * 2
  x0 = int(max(0, x0 - pad_size / 2.))
  y0 = int(max(0, y0 - pad_size / 2.))
  x1 = int(min(image_width, x0 + pad_size))
  y1 = int(min(image_height, y0 + pad_size))
  xy = (x0, y0, x1, y1)
  img = img.copy()
  PIL.ImageDraw.Draw(img).rectangle(xy, (fill, fill, fill))
  return img


def solarize(img: spec.Tensor, threshold: float) -> spec.Tensor:
  img = np.array(img)
  new_img = np.where(img < threshold, img, 255. - img)
  return PIL.Image.fromarray(new_img.astype(np.uint8))


def solarize_add(img: spec.Tensor, addition: int = 0) -> spec.Tensor:
  threshold = 128
  img = np.array(img)
  added_img = img.astype(np.int64) + addition
  added_img = np.clip(added_img, 0, 255).astype(np.uint8)
  new_img = np.where(img < threshold, added_img, img)
  return PIL.Image.fromarray(new_img)


def _apply_op(
    img: spec.Tensor,
    op_name: str,
    magnitude: float,
    interpolation: InterpolationMode,
    fill: Optional[List[float]]) -> spec.Tensor:
  if op_name == 'ShearX':
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
  elif op_name == 'ShearY':
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
  elif op_name == 'TranslateX':
    img = F.affine(
        img,
        angle=0.0,
        translate=[int(magnitude), 0],
        scale=1.0,
        interpolation=interpolation,
        shear=[0.0, 0.0],
        fill=fill,
    )
  elif op_name == 'TranslateY':
    img = F.affine(
        img,
        angle=0.0,
        translate=[0, int(magnitude)],
        scale=1.0,
        interpolation=interpolation,
        shear=[0.0, 0.0],
        fill=fill,
    )
  elif op_name == 'Rotate':
    img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
  elif op_name == 'Brightness':
    img = F.adjust_brightness(img, magnitude)
  elif op_name == 'Color':
    img = F.adjust_saturation(img, magnitude)
  elif op_name == 'Contrast':
    img = F.adjust_contrast(img, magnitude)
  elif op_name == 'Sharpness':
    img = F.adjust_sharpness(img, magnitude)
  elif op_name == 'Posterize':
    img = F.posterize(img, int(magnitude))
  elif op_name == 'Cutout':
    img = cutout(img, magnitude, fill=fill)
  elif op_name == 'SolarizeAdd':
    img = solarize_add(img, int(magnitude))
  elif op_name == 'Solarize':
    img = solarize(img, magnitude)
  elif op_name == 'AutoContrast':
    img = F.autocontrast(img)
  elif op_name == 'Equalize':
    img = F.equalize(img)
  elif op_name == 'Invert':
    img = F.invert(img)
  elif op_name == 'Identity':
    pass
  else:
    raise ValueError(f'The provided operator {op_name} is not recognized.')
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

  def _augmentation_space(self) -> Dict[str, Tuple[spec.Tensor, bool]]:
    return {
        # op_name: (magnitudes, signed)
        'ShearX': (torch.tensor(0.3), True),
        'ShearY': (torch.tensor(0.3), True),
        'TranslateX': (torch.tensor(100), True),
        'TranslateY': (torch.tensor(100), True),
        'Rotate': (torch.tensor(30), True),
        'Brightness': (torch.tensor(1.9), False),
        'Color': (torch.tensor(1.9), False),
        'Contrast': (torch.tensor(1.9), False),
        'Sharpness': (torch.tensor(1.9), False),
        'Posterize': (torch.tensor(4), False),
        'Solarize': (torch.tensor(256), False),
        'SolarizeAdd': (torch.tensor(110), False),
        'AutoContrast': (torch.tensor(0.0), False),
        'Equalize': (torch.tensor(0.0), False),
        'Invert': (torch.tensor(0.0), False),
        'Cutout': (torch.tensor(40.0), False),
    }

  def forward(self, img: spec.Tensor) -> spec.Tensor:
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
