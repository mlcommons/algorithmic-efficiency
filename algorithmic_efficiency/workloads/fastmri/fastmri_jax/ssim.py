"""Structural similarity index calculation in Jax."""

import functools

import jax
import jax.numpy as jnp


def ssim(logits, targets, mean=None, std=None, volume_max=None):
  """Computes example-wise structural similarity for a batch.

  NOTE(dsuo): we use the same (default) arguments to `structural_similarity`
  as in https://arxiv.org/abs/1811.08839.

  Args:
   logits: (batch,) + input.shape float array.
   targets: (batch,) + input.shape float array.
   mean: (batch,) mean of original images.
   std: (batch,) std of original images.
   volume_max: (batch,) of the volume max for the volumes each example came
    from.
  Returns:
    Structural similarity computed per example, shape [batch, ...].
  """
  if volume_max is None:
    volume_max = jnp.ones(logits.shape[0])

  # NOTE(dsuo): `volume_max` can be 0 if we have a padded batch, but this will
  # lead to NaN values in `ssim`.
  volume_max = jnp.where(volume_max == 0, jnp.ones_like(volume_max), volume_max)

  if mean is None:
    mean = jnp.zeros(logits.shape[0])

  if std is None:
    std = jnp.ones(logits.shape[0])

  mean = mean.reshape((-1,) + (1,) * (len(logits.shape) - 1))
  std = std.reshape((-1,) + (1,) * (len(logits.shape) - 1))
  logits = logits * std + mean
  targets = targets * std + mean
  ssims = jax.vmap(structural_similarity)(logits, targets, volume_max)
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
    im1: ndarray Images. Any dimensionality with same shape.
    im2: ndarray Images. Any dimensionality with same shape.
    data_range: float. The data range of the input image (distance
      between minimum and maximum possible values). By default, this is
    win_size: int or None. The side-length of the sliding window used
      in comparison. Must be an odd value. If `gaussian_weights` is True, this
      is ignored and the window size will depend on `sigma`.
      estimated from the image data-type.
    k1: float. Algorithm parameter K1 (see [1]).
    k2: float. Algorithm parameter K2 (see [2]).

  Returns:
    mssim: float
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
  return jnp.mean(s.at[pad:-pad, pad:-pad].get())


def _uniform_filter(im, size=7):

  def conv(im):
    return jnp.convolve(
        jnp.pad(im, pad_width=size // 2, mode='symmetric'),
        jnp.ones(size),
        mode='valid') / size

  im = jax.vmap(conv, (0,))(im)
  im = jax.vmap(conv, (1,))(im)
  return im.T
