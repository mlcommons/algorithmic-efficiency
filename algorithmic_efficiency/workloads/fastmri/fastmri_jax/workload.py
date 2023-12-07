"""FastMRI workload implemented in Jax."""

import functools
import math
from typing import Dict, Optional, Tuple

from flax import jax_utils
import jax
import jax.numpy as jnp

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
import algorithmic_efficiency.random_utils as prng
from algorithmic_efficiency.workloads.fastmri.fastmri_jax import models
from algorithmic_efficiency.workloads.fastmri.fastmri_jax.ssim import ssim
from algorithmic_efficiency.workloads.fastmri.workload import \
    BaseFastMRIWorkload


class FastMRIWorkload(BaseFastMRIWorkload):

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """aux_dropout_rate is unused."""
    del aux_dropout_rate
    fake_batch = jnp.zeros((13, 320, 320))
    self._model = models.UNet(
        num_pool_layers=self.num_pool_layers,
        num_channels=self.num_channels,
        use_tanh=self.use_tanh,
        use_layer_norm=self.use_layer_norm,
        dropout_rate=dropout_rate)

    variables = jax.jit(self._model.init)({'params': rng}, fake_batch)
    params = variables['params']
    self._param_shapes = param_utils.jax_param_shapes(params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
    params = jax_utils.replicate(params)
    return params, None

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key == 'Conv_0'

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
    logits = self._model.apply({'params': params},
                               augmented_and_preprocessed_input_batch['inputs'],
                               rngs={'dropout': rng},
                               train=train)
    return logits, None

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,  # Dense or one-hot labels.
      logits_batch: spec.Tensor,
      mask_batch: Optional[spec.Tensor] = None,
      label_smoothing: float = 0.0) -> Dict[str, spec.Tensor]:  # differentiable
    """Evaluate the (masked) loss function at (label_batch, logits_batch).

    Return {'summed': scalar summed loss, 'n_valid_examples': scalar number of
    valid examples in batch, 'per_example': 1-d array of per-example losses}
    (not synced across devices).
    """
    del label_smoothing
    per_example_losses = jnp.mean(
        jnp.abs(logits_batch - label_batch),
        axis=tuple(range(1, logits_batch.ndim)))
    # mask_batch is assumed to be shape [batch].
    if mask_batch is not None:
      per_example_losses *= mask_batch
      n_valid_examples = mask_batch.sum()
    else:
      n_valid_examples = len(per_example_losses)
    summed_loss = per_example_losses.sum()
    return {
        'summed': summed_loss,
        'n_valid_examples': n_valid_examples,
        'per_example': per_example_losses,
    }

  @functools.partial(
      jax.pmap,
      axis_name='batch',
      in_axes=(None, 0, 0, 0),
      static_broadcasted_argnums=(0,))
  def _eval_model(self,
                  params: spec.Tensor,
                  batch: Dict[str, spec.Tensor],
                  rng: spec.RandomState) -> Dict[str, spec.Tensor]:
    """Return the SSIM and loss as a dict."""
    logits, _ = self.model_fn(
        params,
        batch,
        model_state=None,
        mode=spec.ForwardPassMode.EVAL,
        rng=rng,
        update_batch_norm=False)
    targets = batch['targets']
    weights = batch.get('weights')
    if weights is None:
      weights = jnp.ones(len(logits))
    ssim_vals = ssim(
        logits,
        targets,
        mean=batch['mean'],
        std=batch['std'],
        volume_max=batch['volume_max'])
    ssim_sum = jnp.sum(ssim_vals * weights)
    summed_loss = self.loss_fn(targets, logits, weights)['summed']
    metrics = {
        'ssim': ssim_sum,
        'loss': summed_loss,
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
                           global_step: int = 0) -> Dict[str, float]:
    """Run a full evaluation of the model."""
    del model_state
    del global_step
    data_rng, model_rng = prng.split(rng, 2)
    num_batches = int(math.ceil(num_examples / global_batch_size))
    if split not in self._eval_iters:
      # These iterators repeat indefinitely.
      self._eval_iters[split] = self._build_input_queue(
          data_rng,
          split,
          data_dir,
          global_batch_size=global_batch_size,
          repeat_final_dataset=True,
          num_batches=num_batches)

    total_metrics = {'ssim': 0., 'loss': 0.}
    eval_rngs = prng.split(model_rng, jax.local_device_count())
    for _ in range(num_batches):
      batch = next(self._eval_iters[split])
      # We already sum these metrics across devices inside _eval_model.
      synced_metrics = self._eval_model(params, batch, eval_rngs)
      total_metrics = {
          k: v + synced_metrics[k][0] for k, v in total_metrics.items()
      }
    return {k: float(v.item() / num_examples) for k, v in total_metrics.items()}


class FastMRIModelSizeWorkload(FastMRIWorkload):

  @property
  def num_pool_layers(self) -> bool:
    """Whether or not to use tanh activations in the model."""
    return 3

  @property
  def num_channels(self) -> bool:
    """Whether or not to use tanh activations in the model."""
    return 64


class FastMRITanhWorkload(FastMRIWorkload):

  @property
  def use_tanh(self) -> bool:
    """Whether or not to use tanh activations in the model."""
    return True


class FastMRILayerNormWorkload(FastMRIWorkload):

  @property
  def use_layer_norm(self) -> bool:
    """Whether or not to use tanh activations in the model."""
    return True
