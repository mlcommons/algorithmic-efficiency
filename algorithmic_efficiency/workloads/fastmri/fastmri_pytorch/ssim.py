"""Structural similarity index calculation in PyTorch, ported from Jax."""

import functools

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pad as pad_fn

from algorithmic_efficiency.pytorch_utils import pytorch_setup

DEVICE = pytorch_setup()[2]


def ssim(logits, targets, mean=None, std=None, volume_max=None):
  """Computes example-wise structural similarity for a batch.

  NOTE(dsuo): we use the same (default) arguments to `structural_similarity`
  as in https://arxiv.org/abs/1811.08839.

  Args:
   logits: (batch,) + input.shape float tensor.
   targets: (batch,) + input.shape float tensor.
   mean: (batch,) mean of original images.
   std: (batch,) std of original images.
   volume_max: (batch,) of the volume max for the volumes each example came
    from.
  Returns:
    Structural similarity computed per example, shape [batch, ...].
  """
  if volume_max is None:
    volume_max = torch.ones(logits.shape[0], device=DEVICE)

  # NOTE(dsuo): `volume_max` can be 0 if we have a padded batch, but this will
  # lead to NaN values in `ssim`.
  volume_max = torch.where(volume_max == 0,
                           torch.ones_like(volume_max),
                           volume_max)

  if mean is None:
    mean = torch.zeros(logits.shape[0], device=DEVICE)

  if std is None:
    std = torch.ones(logits.shape[0], device=DEVICE)

  mean = mean.view((-1,) + (1,) * (len(logits.shape) - 1))
  std = std.view((-1,) + (1,) * (len(logits.shape) - 1))
  logits = logits * std + mean
  targets = targets * std + mean
  ssims = torch.vmap(structural_similarity)(logits, targets, volume_max)

  # map out-of-bounds ssims to 1 and -1, the theoretical
  # maximum and minimum values of SSIM.
  ssims = torch.where(ssims > 1, torch.ones_like(ssims), ssims)
  ssims = torch.where(ssims < -1, torch.ones_like(ssims) * -1, ssims)

  return ssims


def structural_similarity(im1,
                          im2,
                          data_range=1.0,
                          win_size=7,
                          k1=0.01,
                          k2=0.03):
  """Compute the mean structural similarity index between two images.

  NOTE(dsuo): modified from skimage.metrics.structural_similarity.

  Args:
    im1: Image tensor. Any dimensionality with same shape.
    im2: Image tensor. Any dimensionality with same shape.
    data_range: float. The data range of the input image (distance
      between minimum and maximum possible values). By default, this is
    win_size: int or None. The side-length of the sliding window used
      in comparison. Must be an odd value. If `gaussian_weights` is True, this
      is ignored and the window size will depend on `sigma`.
      estimated from the image data-type.
    k1: float. Algorithm parameter K1 (see [1]).
    k2: float. Algorithm parameter K2 (see [2]).

  Returns:
    mssim: Scalar float tensor.
        The mean structural similarity index over the image.

  References
    [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
      (2004). Image quality assessment: From error visibility to
      structural similarity. IEEE Transactions on Image Processing,
      13, 600-612.
      https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
      :DOI:`10.1109/TIP.2003.819861`
  """
  filter_func = functools.partial(_uniform_filter, size=win_size)

  num_points = win_size**len(im1.shape)

  # filter has already normalized by num_points
  cov_norm = num_points / (num_points - 1)  # sample covariance

  # compute (weighted) means
  ux = filter_func(im1)
  uy = filter_func(im2)

  # compute (weighted) variances and covariances
  uxx = filter_func(im1 * im1)
  uyy = filter_func(im2 * im2)
  uxy = filter_func(im1 * im2)
  vx = cov_norm * (uxx - ux * ux)
  vy = cov_norm * (uyy - uy * uy)
  vxy = cov_norm * (uxy - ux * uy)

  c1 = (k1 * data_range)**2
  c2 = (k2 * data_range)**2

  a1 = 2 * ux * uy + c1
  a2 = 2 * vxy + c2
  b1 = ux**2 + uy**2 + c1
  b2 = vx + vy + c2

  d = b1 * b2
  s = (a1 * a2) / d

  # to avoid edge effects will ignore filter radius strip around edges
  pad = (win_size - 1) // 2

  # compute (weighted) mean of ssim.
  return torch.mean(s[pad:-pad, pad:-pad])


def _uniform_filter(im, size=7):
  pad_size = size // 2

  def conv(im):
    # This function does not seem to work with only two dimensions.
    padded_im = pad_fn(im.unsqueeze(0), pad_size, padding_mode='symmetric')
    # Remove the first dim and the padding from the second dim.
    padded_im = padded_im[0, pad_size:-pad_size]
    filters = torch.ones(1, 1, size, dtype=padded_im.dtype, device=DEVICE)
    # Add additional dimension for the number of channels.
    return F.conv1d(padded_im.unsqueeze(1), filters).squeeze(1) / size

  im = conv(im)
  im = conv(im.T)
  return im.T
