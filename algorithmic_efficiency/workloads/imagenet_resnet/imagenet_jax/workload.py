"""ImageNet workload implemented in Jax.

Forked from the Flax ImageNet Example v0.3.3
https://github.com/google/flax/tree/v0.3.3/examples/imagenet.
"""

import functools
import itertools
import math
from typing import Dict, Iterator, Optional, Tuple

from flax import jax_utils
from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.imagenet_resnet import imagenet_v2
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax import \
    input_pipeline
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax import \
    models
from algorithmic_efficiency.workloads.imagenet_resnet.workload import \
    BaseImagenetResNetWorkload


class ImagenetResNetWorkload(BaseImagenetResNetWorkload):

  def _build_dataset(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      global_batch_size: int,
      cache: Optional[bool] = None,
      repeat_final_dataset: Optional[bool] = None,
      use_mixup: bool = False,
      use_randaug: bool = False) -> Iterator[Dict[str, spec.Tensor]]:
    if split == 'test':
      np_iter = imagenet_v2.get_imagenet_v2_iter(
          data_dir,
          global_batch_size,
          mean_rgb=self.train_mean,
          stddev_rgb=self.train_stddev,
          image_size=self.center_crop_size,
          resize_size=self.resize_size)
      return itertools.cycle(np_iter)

    ds_builder = tfds.builder('imagenet2012:5.1.0', data_dir=data_dir)
    train = split == 'train'
    ds = input_pipeline.create_input_iter(
        split,
        ds_builder,
        data_rng,
        global_batch_size,
        self.train_mean,
        self.train_stddev,
        self.center_crop_size,
        self.resize_size,
        self.aspect_ratio_range,
        self.scale_ratio_range,
        train=train,
        cache=not train if cache is None else cache,
        repeat_final_dataset=repeat_final_dataset,
        use_mixup=use_mixup,
        mixup_alpha=0.2,
        use_randaug=use_randaug)
    return ds

  def sync_batch_stats(
      self, model_state: spec.ModelAuxiliaryState) -> spec.ModelAuxiliaryState:
    """Sync the batch statistics across replicas."""
    # An axis_name is passed to pmap which can then be used by pmean.
    # In this case each device has its own version of the batch statistics and
    # we average them.
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
    model_cls = getattr(models, 'ResNet50')

    if self.use_silu and self.use_gelu:
      raise RuntimeError('Cannot use both GELU and SiLU activations.')
    if self.use_silu:
      act_fnc = nn.silu
    elif self.use_gelu:
      act_fnc = nn.gelu
    else:
      act_fnc = nn.relu

    model = model_cls(
        num_classes=self._num_classes,
        act=act_fnc,
        bn_init_scale=self.bn_init_scale,
        dtype=jnp.float32)
    self._model = model
    input_shape = (1, 224, 224, 3)
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

  @functools.partial(
      jax.pmap,
      axis_name='batch',
      in_axes=(None, 0, 0, 0, 0),
      static_broadcasted_argnums=(0,))
  def _eval_model(self,
                  params: spec.ParameterContainer,
                  batch: Dict[str, spec.Tensor],
                  model_state: spec.ModelAuxiliaryState,
                  rng: spec.RandomState) -> Dict[str, spec.Tensor]:
    logits, _ = self.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.EVAL,
        rng=rng,
        update_batch_norm=False)
    weights = batch.get('weights')
    return self._compute_metrics(logits, batch['targets'], weights)

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool,
      use_running_average_bn: Optional[bool] = None) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del mode
    del rng
    variables = {'params': params, **model_state}
    if update_batch_norm:
      logits, new_model_state = self._model.apply(
          variables,
          augmented_and_preprocessed_input_batch['inputs'],
          update_batch_norm=update_batch_norm,
          mutable=['batch_stats'],
          use_running_average_bn=use_running_average_bn)
      return logits, new_model_state
    else:
      logits = self._model.apply(
          variables,
          augmented_and_preprocessed_input_batch['inputs'],
          update_batch_norm=update_batch_norm,
          mutable=False,
          use_running_average_bn=use_running_average_bn)
      return logits, model_state

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
    if label_batch.shape[-1] != self._num_classes:
      one_hot_labels = jax.nn.one_hot(
          label_batch, num_classes=self._num_classes)
    else:
      one_hot_labels = label_batch
    smoothed_labels = optax.smooth_labels(one_hot_labels, label_smoothing)
    per_example_losses = -jnp.sum(
        smoothed_labels * jax.nn.log_softmax(logits_batch, axis=-1), axis=-1)
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

  def _compute_metrics(self,
                       logits: spec.Tensor,
                       labels: spec.Tensor,
                       weights: spec.Tensor) -> Dict[str, spec.Tensor]:
    if weights is None:
      weights = jnp.ones(len(logits))
    summed_loss = self.loss_fn(labels, logits, weights)['summed']
    # not accuracy, but nr. of correct predictions
    accuracy = jnp.sum((jnp.argmax(logits, -1) == labels) * weights)
    metrics = {
        'loss': summed_loss,
        'accuracy': accuracy,
    }
    metrics = lax.psum(metrics, axis_name='batch')
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
    del global_step
    if model_state is not None:
      # Sync batch statistics across replicas before evaluating.
      model_state = self.sync_batch_stats(model_state)
    num_batches = int(math.ceil(num_examples / global_batch_size))
    data_rng, eval_rng = prng.split(rng, 2)
    # We already repeat the dataset indefinitely in tf.data.
    if split not in self._eval_iters:
      self._eval_iters[split] = self._build_input_queue(
          data_rng,
          split=split,
          global_batch_size=global_batch_size,
          data_dir=data_dir,
          cache=True,
          repeat_final_dataset=True,
          num_batches=num_batches)

    eval_metrics = {}
    for bi in range(num_batches):
      eval_rng = prng.fold_in(eval_rng, bi)
      step_eval_rngs = prng.split(eval_rng, jax.local_device_count())
      batch = next(self._eval_iters[split])
      # We already average these metrics across devices inside _compute_metrics.
      synced_metrics = self._eval_model(params,
                                        batch,
                                        model_state,
                                        step_eval_rngs)
      for metric_name, metric_value in synced_metrics.items():
        if metric_name not in eval_metrics:
          eval_metrics[metric_name] = 0.0
        eval_metrics[metric_name] += metric_value

    eval_metrics = jax.tree_map(lambda x: float(x[0] / num_examples),
                                eval_metrics)
    return eval_metrics


class ImagenetResNetSiLUWorkload(ImagenetResNetWorkload):

  @property
  def use_silu(self) -> bool:
    return True

  @property
  def validation_target_value(self) -> float:
    return 0.75445

  @property
  def test_target_value(self) -> float:
    return 0.6323


class ImagenetResNetGELUWorkload(ImagenetResNetWorkload):

  @property
  def use_gelu(self) -> bool:
    return True

  @property
  def validation_target_value(self) -> float:
    return 0.76765

  @property
  def test_target_value(self) -> float:
    return 0.6519


class ImagenetResNetLargeBNScaleWorkload(ImagenetResNetWorkload):

  @property
  def bn_init_scale(self) -> float:
    return 8.0

  @property
  def validation_target_value(self) -> float:
    return 0.76526

  @property
  def test_target_value(self) -> float:
    return 0.6423
