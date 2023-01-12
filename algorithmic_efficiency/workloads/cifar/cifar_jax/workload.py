"""CIFAR workload implemented in Jax."""

import functools
from typing import Dict, Iterator, Optional, Tuple

from flax import jax_utils
from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.cifar.cifar_jax.input_pipeline import \
    create_input_iter
from algorithmic_efficiency.workloads.cifar.workload import BaseCifarWorkload
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax import \
    models


class CifarWorkload(BaseCifarWorkload):

  def _build_cifar_dataset(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      batch_size: int,
      cache: Optional[bool] = None,
      repeat_final_dataset: Optional[bool] = None
  ) -> Iterator[Dict[str, spec.Tensor]]:
    ds_builder = tfds.builder('cifar10:3.0.2', data_dir=data_dir)
    ds_builder.download_and_prepare()
    train = split == 'train'
    assert self.num_train_examples + self.num_validation_examples == 50000
    if split in ['train', 'eval_train']:
      split = f'train[:{self.num_train_examples}]'
    elif split == 'validation':
      split = f'train[{self.num_train_examples}:]'
    ds = create_input_iter(
        split,
        ds_builder,
        data_rng,
        batch_size,
        self.train_mean,
        self.train_stddev,
        self.center_crop_size,
        self.aspect_ratio_range,
        self.scale_ratio_range,
        train=train,
        cache=not train if cache is None else cache,
        repeat_final_dataset=repeat_final_dataset)
    return ds

  def _build_input_queue(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      global_batch_size: int,
      cache: Optional[bool] = None,
      repeat_final_dataset: Optional[bool] = None,
      num_batches: Optional[int] = None) -> Iterator[Dict[str, spec.Tensor]]:
    del num_batches
    return self._build_cifar_dataset(data_rng,
                                     split,
                                     data_dir,
                                     global_batch_size,
                                     cache,
                                     repeat_final_dataset)

  def sync_batch_stats(
      self, model_state: spec.ModelAuxiliaryState) -> spec.ModelAuxiliaryState:
    """Sync the batch statistics across replicas."""
    # An axis_name is passed to pmap which can then be used by pmean.
    # In this case each device has its own version of the batch statistics
    # and we average them.
    avg_fn = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')
    new_model_state = model_state.copy(
        {'batch_stats': avg_fn(model_state['batch_stats'])})
    return new_model_state

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """Dropout is unused."""
    del dropout_rate
    del aux_dropout_rate
    model_cls = getattr(models, 'ResNet18')
    model = model_cls(num_classes=10, dtype=jnp.float32)
    self._model = model
    input_shape = (1, 32, 32, 3)
    variables = jax.jit(model.init)({'params': rng},
                                    jnp.ones(input_shape, model.dtype))
    model_state, params = variables.pop('params')
    self._param_shapes = param_utils.jax_param_shapes(params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
    model_state = jax_utils.replicate(model_state)
    params = jax_utils.replicate(params)
    return params, model_state

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key == 'Dense_0'

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del mode
    del rng
    variables = {'params': params, **model_state}
    if update_batch_norm:
      logits, new_model_state = self._model.apply(
          variables,
          augmented_and_preprocessed_input_batch['inputs'],
          update_batch_norm=update_batch_norm,
          mutable=['batch_stats'])
      return logits, new_model_state
    else:
      logits = self._model.apply(
          variables,
          augmented_and_preprocessed_input_batch['inputs'],
          update_batch_norm=update_batch_norm,
          mutable=False)
      return logits, model_state

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(self,
              label_batch: spec.Tensor,
              logits_batch: spec.Tensor,
              mask_batch: Optional[spec.Tensor] = None,
              label_smoothing: float = 0.0) -> Tuple[spec.Tensor, spec.Tensor]:
    """Return (correct scalar average loss, 1-d array of per-example losses)."""
    one_hot_targets = jax.nn.one_hot(label_batch, 10)
    smoothed_targets = optax.smooth_labels(one_hot_targets, label_smoothing)
    per_example_losses = -jnp.sum(
        smoothed_targets * nn.log_softmax(logits_batch), axis=-1)
    # `mask_batch` is assumed to be shape [batch].
    if mask_batch is not None:
      per_example_losses *= mask_batch
      n_valid_examples = mask_batch.sum()
    else:
      n_valid_examples = len(per_example_losses)
    summed_loss = per_example_losses.sum()
    return summed_loss / n_valid_examples, per_example_losses

  def _compute_metrics(self,
                       logits: spec.Tensor,
                       labels: spec.Tensor,
                       weights: spec.Tensor) -> Dict[str, spec.Tensor]:
    _, per_example_losses = self.loss_fn(labels, logits, weights)
    loss = jnp.sum(per_example_losses)
    # Number of correct predictions.
    accuracy = jnp.sum((jnp.argmax(logits, -1) == labels) * weights)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    metrics = lax.psum(metrics, axis_name='batch')
    return metrics

  @functools.partial(
      jax.pmap,
      axis_name='batch',
      in_axes=(None, 0, 0, 0, None),
      static_broadcasted_argnums=(0,))
  def _eval_model(
      self,
      params: spec.ParameterContainer,
      batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      rng: spec.RandomState) -> Dict[spec.Tensor, spec.ModelAuxiliaryState]:
    """Return the mean accuracy and loss as a dict."""
    logits, _ = self.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.EVAL,
        rng,
        update_batch_norm=False)
    weights = batch.get('weights')
    if weights is None:
      weights = jnp.ones(len(logits))
    return self._compute_metrics(logits, batch['targets'], weights)
