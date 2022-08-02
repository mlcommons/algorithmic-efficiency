"""FastMRI workload implemented in Jax."""

import functools
import math
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from skimage.metrics import structural_similarity

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
import algorithmic_efficiency.random_utils as prng
from algorithmic_efficiency.workloads.fastmri.fastmri_jax import input_pipeline
from algorithmic_efficiency.workloads.fastmri.fastmri_jax import models
from algorithmic_efficiency.workloads.fastmri.workload import \
    BaseFastMRIWorkload


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

  def __init__(self):
    super().__init__()
    self._param_types = None
    self._eval_iters = {}
    self._model = models.UNet()

  @property
  def param_shapes(self):
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    return self._param_shapes

  @property
  def model_params_types(self):
    """The shapes of the parameters in the workload model."""
    if self._param_types is None:
      self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    return self._param_types

  def build_input_queue(self,
                        data_rng,
                        split: str,
                        data_dir: str,
                        global_batch_size: int,
                        cache: Optional[bool] = None,
                        repeat_final_dataset: Optional[bool] = None,
                        num_batches: Optional[int] = None):
    del cache
    per_host_batch_size = global_batch_size // jax.num_local_devices()
    return input_pipeline.load_fastmri_split(
        per_host_batch_size, split, data_dir, data_rng, num_batches,
        repeat_final_dataset)

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    fake_batch = None
    variables = jax.jit(self._model.init)({'params': rng}, fake_batch)
    params = variables['params']
    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      params)
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
    del rng
    del update_batch_norm
    train = mode == spec.ForwardPassMode.TRAIN
    logits = self._model.apply({'params': params},
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
  def loss_fn(self, targets_batch: spec.Tensor,
              outputs_batch: spec.Tensor) -> spec.Tensor:  # differentiable
    return jnp.abs(outputs_batch - targets_batch).mean(
        axis=tuple(range(1, outputs_batch.ndim)))

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
    loss = jnp.sum(self.loss_fn(outputs, batch['targets']))
    metrics = {
        'ssim': ssim_vals, 'loss': loss, 'weight': jnp.sum(batch['weights']),
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
                           data_dir: str):
    """Run a full evaluation of the model."""
    del model_state
    data_rng, model_rng = prng.split(rng, 2)
    if split not in self._eval_iters:
      # These iterators repeat indefinitely.
      self._eval_iters[split] = self.build_input_queue(
          data_rng, split, data_dir, global_batch_size=global_batch_size,
          repeat_final_dataset=True)

    total_metrics = {'ssim': 0., 'loss': 0., 'weight': 0.}
    num_batches = int(math.ceil(num_examples / global_batch_size))
    eval_rngs = prng.split(model_rng, jax.num_local_devices())
    for _ in range(num_batches):
      batch = next(self._eval_iters[split])
      # We already average these metrics across devices inside _compute_metrics.
      synced_metrics = self._eval_model(params, batch, eval_rngs)
      total_metrics = {
          k: v + synced_metrics[k] for k, v in total_metrics.items()
      }
    num_examples = total_metrics.pop('weight')
    return {k: float(v.item() / num_examples) for k, v in total_metrics.items()}
