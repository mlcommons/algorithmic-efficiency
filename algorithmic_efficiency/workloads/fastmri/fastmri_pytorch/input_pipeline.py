"""Data pipeline for FastMRI dataset.

Modified from https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data.
"""

import logging
import os
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn
import contextlib
from typing import Optional, Sequence, Tuple, Union
from algorithmic_efficiency.workloads.fastmri.fastmri_pytorch import fftc
import numpy as np
import torch
import h5py
import numpy as np
import pandas as pd
import requests
import torch


def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
  """
  ElementTree query function.
  This can be used to query an xml document via ElementTree. It uses qlist
  for nested queries.
  Args:
      root: Root of the xml to search through.
      qlist: A list of strings for nested searches, e.g. ["Encoding",
          "matrixSize"]
      namespace: Optional; xml namespace to prepend query.
  Returns:
      The retrieved data as a string.
  """
  s = "."
  prefix = "ismrmrd_namespace"

  ns = {prefix: namespace}

  for el in qlist:
    s = s + f"//{prefix}:{el}"

  value = root.find(s, ns)
  if value is None:
    raise RuntimeError("Element not found")

  return str(value.text)

def to_tensor(data: np.ndarray) -> torch.Tensor:
  """
  Convert numpy array to PyTorch tensor.
  For complex arrays, the real and imaginary parts are stacked along the last
  dimension.
  Args:
      data: Input numpy array.
  Returns:
      PyTorch version of data.
  """
  if np.iscomplexobj(data):
    data = np.stack((data.real, data.imag), axis=-1)

  return torch.from_numpy(data)

def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
  """
  Subsample given k-space by multiplying with a mask.
  Args:
      data: The input k-space data. This should have at least 3 dimensions,
          where dimensions -3 and -2 are the spatial dimensions, and the
          final dimension has size 2 (for complex values).
      mask_func: A function that takes a shape (tuple of ints) and a random
          number seed and returns a mask.
      seed: Seed for the random number generator.
      padding: Padding value to apply for mask.
  Returns:
      tuple containing:
          masked data: Subsampled k-space data.
          mask: The generated mask.
          num_low_frequencies: The number of low-resolution frequency samples
              in the mask.
  """
  shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
  mask, num_low_frequencies = mask_func(shape, offset, seed)
  if padding is not None:
    mask[:, :, : padding[0]] = 0
    mask[:, :, padding[1]:] = 0  # padding value inclusive on right of zeros

  masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

  return masked_data, mask, num_low_frequencies

  def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.
    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.
    Args:
        data: Input numpy array.
    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
      data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
  """
  Converts a complex torch tensor to numpy array.
  Args:
      data: Input data to be converted to numpy.
  Returns:
      Complex numpy version of data.
  """
  return torch.view_as_complex(data).numpy()


def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
  """
  Subsample given k-space by multiplying with a mask.
  Args:
      data: The input k-space data. This should have at least 3 dimensions,
          where dimensions -3 and -2 are the spatial dimensions, and the
          final dimension has size 2 (for complex values).
      mask_func: A function that takes a shape (tuple of ints) and a random
          number seed and returns a mask.
      seed: Seed for the random number generator.
      padding: Padding value to apply for mask.
  Returns:
      tuple containing:
          masked data: Subsampled k-space data.
          mask: The generated mask.
          num_low_frequencies: The number of low-resolution frequency samples
              in the mask.
  """
  shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
  mask, num_low_frequencies = mask_func(shape, offset, seed)
  if padding is not None:
    mask[:, :, : padding[0]] = 0
    mask[:, :, padding[1]:] = 0  # padding value inclusive on right of zeros

  masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

  return masked_data, mask, num_low_frequencies


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
  """
  Initializes a mask with the center filled in.
  Args:
      mask_from: Part of center to start filling.
      mask_to: Part of center to end filling.
  Returns:
      A mask with the center filled.
  """
  mask = torch.zeros_like(x)
  mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

  return mask


def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
  """
  Initializes a mask with the center filled in.
  Can operate with different masks for each batch element.
  Args:
      mask_from: Part of center to start filling.
      mask_to: Part of center to end filling.
  Returns:
      A mask with the center filled.
  """
  if not mask_from.shape == mask_to.shape:
    raise ValueError("mask_from and mask_to must match shapes.")
  if not mask_from.ndim == 1:
    raise ValueError("mask_from and mask_to must have 1 dimension.")
  if not mask_from.shape[0] == 1:
    if (not x.shape[0] == mask_from.shape[0]) or (
        not x.shape[0] == mask_to.shape[0]
    ):
      raise ValueError("mask_from and mask_to must have batch_size length.")

  if mask_from.shape[0] == 1:
    mask = mask_center(x, int(mask_from), int(mask_to))
  else:
    mask = torch.zeros_like(x)
    for i, (start, end) in enumerate(zip(mask_from, mask_to)):
      mask[i, :, :, start:end] = x[i, :, :, start:end]

  return mask


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
  """
  Apply a center crop to the input real image or batch of real images.
  Args:
      data: The input tensor to be center cropped. It should
          have at least 2 dimensions and the cropping is applied along the
          last two dimensions.
      shape: The output shape. The shape should be smaller
          than the corresponding dimensions of data.
  Returns:
      The center cropped image.
  """
  if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
    raise ValueError("Invalid shapes.")

  w_from = (data.shape[-2] - shape[0]) // 2
  h_from = (data.shape[-1] - shape[1]) // 2
  w_to = w_from + shape[0]
  h_to = h_from + shape[1]

  return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
  """
  Apply a center crop to the input image or batch of complex images.
  Args:
      data: The complex input tensor to be center cropped. It should have at
          least 3 dimensions and the cropping is applied along dimensions -3
          and -2 and the last dimensions should have a size of 2.
      shape: The output shape. The shape should be smaller than the
          corresponding dimensions of data.
  Returns:
      The center cropped image
  """
  if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
    raise ValueError("Invalid shapes.")

  w_from = (data.shape[-3] - shape[0]) // 2
  h_from = (data.shape[-2] - shape[1]) // 2
  w_to = w_from + shape[0]
  h_to = h_from + shape[1]

  return data[..., w_from:w_to, h_from:h_to, :]


def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Apply a center crop on the larger image to the size of the smaller.
  The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
  dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
  be a mixture of the two.
  Args:
      x: The first image.
      y: The second image.
  Returns:
      tuple of tensors x and y, each cropped to the minimim size.
  """
  smallest_width = min(x.shape[-1], y.shape[-1])
  smallest_height = min(x.shape[-2], y.shape[-2])
  x = center_crop(x, (smallest_height, smallest_width))
  y = center_crop(y, (smallest_height, smallest_width))

  return x, y


def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
  """
  Normalize the given tensor.
  Applies the formula (data - mean) / (stddev + eps).
  Args:
      data: Input data to be normalized.
      mean: Mean value.
      stddev: Standard deviation.
      eps: Added to stddev to prevent dividing by zero.
  Returns:
      Normalized tensor.
  """
  return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """
  Normalize the given tensor  with instance norm/
  Applies the formula (data - mean) / (stddev + eps), where mean and stddev
  are computed from the data itself.
  Args:
      data: Input data to be normalized
      eps: Added to stddev to prevent dividing by zero.
  Returns:
      torch.Tensor: Normalized tensor
  """
  mean = data.mean()
  std = data.std()

  return normalize(data, mean, std, eps), mean, std


def _process_example(kspace, mask_func, mask, target, attrs, fname, slice_num):
  kspace_torch = to_tensor(kspace)

  # check for max value
  max_value = attrs["max"] if "max" in attrs.keys() else 0.0

  # apply mask
  if mask_func:
    seed = tuple(map(ord, fname))
    # we only need first element, which is k-space after masking
    masked_kspace = apply_mask(kspace_torch, mask_func, seed=seed)[0]
  else:
    masked_kspace = kspace_torch

  # inverse Fourier transform to get zero filled solution
  image = ifft2c(masked_kspace)

  # crop input to correct size
  if target is not None:
    crop_size = (target.shape[-2], target.shape[-1])
  else:
    crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

  # check for FLAIR 203
  if image.shape[-2] < crop_size[1]:
    crop_size = (image.shape[-2], image.shape[-2])

  image = complex_center_crop(image, crop_size)

  # absolute value
  image = fastmri.complex_abs(image)


  # normalize input
  image, mean, std = normalize_instance(image, eps=1e-11)
  image = image.clamp(-6, 6)

  # normalize target
  if target is not None:
    target_torch = to_tensor(target)
    target_torch = center_crop(target_torch, crop_size)
    target_torch = normalize(target_torch, mean, std, eps=1e-11)
    target_torch = target_torch.clamp(-6, 6)
  else:
    target_torch = torch.Tensor([0])

  return {'inputs': image, 'targets': target_torch, 'volume_max': max_value}


class SliceDataset(torch.utils.data.Dataset):
  """
  A PyTorch Dataset that provides access to MR image slices.
  """

  def __init__(
      self,
      root: Union[str, Path, os.PathLike],
      transform: Optional[Callable] = None,
      use_dataset_cache: bool = False,
      sample_rate: Optional[float] = None,
      volume_sample_rate: Optional[float] = None,
      dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
      num_cols: Optional[Tuple[int]] = None,
  ):
    """
    Args:
        root: Path to the dataset.
        challenge: "singlecoil" or "multicoil" depending on which challenge
            to use.
        transform: Optional; A callable object that pre-processes the raw
            data into appropriate form. The transform function should take
            'kspace', 'target', 'attributes', 'filename', and 'slice' as
            inputs. 'target' may be null for test data.
        use_dataset_cache: Whether to cache dataset metadata. This is very
            useful for large datasets like the brain data.
        sample_rate: Optional; A float between 0 and 1. This controls what fraction
            of the slices should be loaded. Defaults to 1 if no value is given.
            When creating a sampled dataset either set sample_rate (sample by slices)
            or volume_sample_rate (sample by volumes) but not both.
        volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
            of the volumes should be loaded. Defaults to 1 if no value is given.
            When creating a sampled dataset either set sample_rate (sample by slices)
            or volume_sample_rate (sample by volumes) but not both.
        dataset_cache_file: Optional; A file in which to cache dataset
            information for faster load times.
        num_cols: Optional; If provided, only slices with the desired
            number of columns will be considered.
    """
    if sample_rate is not None and volume_sample_rate is not None:
      raise ValueError(
        "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
      )

    self.dataset_cache_file = Path(dataset_cache_file)

    self.transform = transform
    self.recons_key = "reconstruction_esc"
    self.examples = []

    # set default sampling mode if none given
    if sample_rate is None:
      sample_rate = 1.0
    if volume_sample_rate is None:
      volume_sample_rate = 1.0

    # load dataset cache if we have and user wants to use it
    if self.dataset_cache_file.exists() and use_dataset_cache:
      with open(self.dataset_cache_file, "rb") as f:
        dataset_cache = pickle.load(f)
    else:
      dataset_cache = {}

    # check if our dataset is in the cache
    # if there, use that metadata, if not, then regenerate the metadata
    if dataset_cache.get(root) is None or not use_dataset_cache:
      files = list(Path(root).iterdir())
      for fname in sorted(files):
        metadata, num_slices = self._retrieve_metadata(fname)

        self.examples += [
          (fname, slice_ind, metadata) for slice_ind in range(num_slices)
        ]

      if dataset_cache.get(root) is None and use_dataset_cache:
        dataset_cache[root] = self.examples
        logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
        with open(self.dataset_cache_file, "wb") as cache_f:
          pickle.dump(dataset_cache, cache_f)
    else:
      logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
      self.examples = dataset_cache[root]

    # subsample if desired
    if sample_rate < 1.0:  # sample by slice
      random.shuffle(self.examples)
      num_examples = round(len(self.examples) * sample_rate)
      self.examples = self.examples[:num_examples]
    elif volume_sample_rate < 1.0:  # sample by volume
      vol_names = sorted(list(set([f[0].stem for f in self.examples])))
      random.shuffle(vol_names)
      num_volumes = round(len(vol_names) * volume_sample_rate)
      sampled_vols = vol_names[:num_volumes]
      self.examples = [
        example for example in self.examples if example[0].stem in sampled_vols
      ]

    if num_cols:
      self.examples = [
        ex
        for ex in self.examples
        if ex[2]["encoding_size"][1] in num_cols  # type: ignore
      ]

  def _retrieve_metadata(self, fname):
    with h5py.File(fname, "r") as hf:
      et_root = etree.fromstring(hf["ismrmrd_header"][()])

      enc = ["encoding", "encodedSpace", "matrixSize"]
      enc_size = (
        int(et_query(et_root, enc + ["x"])),
        int(et_query(et_root, enc + ["y"])),
        int(et_query(et_root, enc + ["z"])),
      )
      rec = ["encoding", "reconSpace", "matrixSize"]
      recon_size = (
        int(et_query(et_root, rec + ["x"])),
        int(et_query(et_root, rec + ["y"])),
        int(et_query(et_root, rec + ["z"])),
      )

      lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
      enc_limits_center = int(et_query(et_root, lims + ["center"]))
      enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

      padding_left = enc_size[1] // 2 - enc_limits_center
      padding_right = padding_left + enc_limits_max

      num_slices = hf["kspace"].shape[0]

    metadata = {
      "padding_left": padding_left,
      "padding_right": padding_right,
      "encoding_size": enc_size,
      "recon_size": recon_size,
    }

    return metadata, num_slices

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, i: int):
    fname, dataslice, metadata = self.examples[i]

    with h5py.File(fname, "r") as hf:
      kspace = hf["kspace"][dataslice]

      mask = np.asarray(hf["mask"]) if "mask" in hf else None

      target = hf[self.recons_key][dataslice] if self.recons_key in hf else None

      attrs = dict(hf.attrs)
      attrs.update(metadata)

    if self.transform is None:
      sample = (kspace, mask, target, attrs, fname.name, dataslice)
    else:
      sample = self.transform(kspace, mask, target, attrs, fname.name, dataslice)

    return sample


a = SliceDataset("/Users/juhanbae/Downloads/singlecoil_test_v2")
print(help(a))
