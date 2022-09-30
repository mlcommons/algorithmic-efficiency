"""FastMRI workload implemented in Jax."""

import functools
import math
from typing import Dict, Tuple

from flax import jax_utils
import jax
import jax.numpy as jnp

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
import algorithmic_efficiency.random_utils as prng
from algorithmic_efficiency.workloads.fastmri.fastmri_jax import models
from algorithmic_efficiency.workloads.fastmri.workload import \
    BaseFastMRIWorkload


def _uniform_filter(im, size=7):

  def conv(im):
    return jnp.convolve(
        jnp.pad(im, pad_width=size // 2, mode='symmetric'),
        jnp.ones(size),
        mode='valid') / size

  im = jax.vmap(conv, (0,))(im)
  im = jax.vmap(conv, (1,))(im)
  return im.T


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


class FastMRIWorkload(BaseFastMRIWorkload):

  @property
  def model_params_types(self):
    """The shapes of the parameters in the workload model."""
    if self._param_types is None:
      self._param_types = param_utils.jax_param_types(self._param_shapes)
    return self._param_types

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    fake_batch = jnp.zeros((13, 320, 320))
    variables = jax.jit(models.UNet().init)({'params': rng}, fake_batch)
    params = variables['params']
    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      params)
    params = jax_utils.replicate(params)
    return params, None

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del update_batch_norm
    train = mode == spec.ForwardPassMode.TRAIN
    logits = models.UNet().apply(
        {'params': params},
        augmented_and_preprocessed_input_batch['inputs'],
        rngs={'dropout': rng},
        train=train)
    return logits, None

  def output_activation_fn(self,
                           logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:

    activation_fn = {
        spec.LossType.SOFTMAX_CROSS_ENTROPY: jax.nn.softmax,
        spec.LossType.SIGMOID_CROSS_ENTROPY: jax.nn.sigmoid,
        spec.LossType.MEAN_SQUARED_ERROR: lambda z: z,
        spec.LossType.MEAN_ABSOLUTE_ERROR: lambda z: z
    }
    return activation_fn[loss_type](logits_batch)

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(self, label_batch: spec.Tensor,
              outputs_batch: spec.Tensor) -> spec.Tensor:  # differentiable
    return jnp.abs(outputs_batch -
                   label_batch).mean(axis=tuple(range(1, outputs_batch.ndim)))

  @functools.partial(
      jax.pmap,
      axis_name='batch',
      in_axes=(None, 0, 0, 0),
      static_broadcasted_argnums=(0,))
  def _eval_model(self, params, batch, rng):
    """Return the SSIM and loss as a dict."""
    outputs, _ = self.model_fn(
        params,
        batch,
        model_state=None,
        mode=spec.ForwardPassMode.EVAL,
        rng=rng,
        update_batch_norm=False)
    ssim_vals = ssim(
        batch['targets'],
        outputs,
        mean=batch['mean'],
        std=batch['std'],
        volume_max=batch['volume_max'])
    ssim_sum = jnp.sum(ssim_vals)
    loss = jnp.sum(self.loss_fn(outputs, batch['targets']))
    metrics = {
        'ssim': ssim_sum,
        'loss': loss,
        'weight': jnp.sum(batch['weights']),
    }
    metrics = jax.lax.psum(metrics, axis_name='batch')
    return metrics

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str,
                           global_step: int = 0):
    """Run a full evaluation of the model."""
    del model_state
    del global_step
    data_rng, model_rng = prng.split(rng, 2)
    if split not in self._eval_iters:
      # These iterators repeat indefinitely.
      self._eval_iters[split] = self.build_input_queue(
          data_rng,
          split,
          data_dir,
          global_batch_size=global_batch_size,
          repeat_final_dataset=True)

    total_metrics = {'ssim': 0., 'loss': 0., 'weight': 0.}
    num_batches = int(math.ceil(num_examples / global_batch_size))
    eval_rngs = prng.split(model_rng, jax.local_device_count())
    for _ in range(num_batches):
      batch = next(self._eval_iters[split])
      # We already sum these metrics across devices inside _compute_metrics.
      synced_metrics = self._eval_model(params, batch, eval_rngs)
      total_metrics = {
          k: v + synced_metrics[k][0] for k, v in total_metrics.items()
      }
    num_examples = total_metrics.pop('weight')
    return {k: float(v.item() / num_examples) for k, v in total_metrics.items()}
