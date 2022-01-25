"""ImageNet workload implemented in Jax.

python3 submission_runner.py \
    --workload=imagenet_jax \
    --submission_path=workloads/imagenet/imagenet_jax/submission.py \
    --num_tuning_trials=1
"""
import functools
from typing import Tuple
import optax

import tensorflow as tf
# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make it
# unavailable to JAX.
tf.config.experimental.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from flax import jax_utils

import spec
import random_utils as prng
from workloads.imagenet.workload import ImagenetWorkload
from . import input_pipeline
from . import models


class ImagenetWorkload(ImagenetWorkload):
  def __init__(self):
    super().__init__()
    self._param_shapes = None
    self.epoch_metrics = []

  @property
  def num_train_examples(self):
    return 1271167

  @property
  def num_eval_examples(self):
    return 100000

  def _build_dataset(self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      batch_size):
    if batch_size % jax.device_count() > 0:
      raise ValueError('Batch size must be divisible by the number of devices')
    ds_builder = tfds.builder('imagenet2012:5.*.*')
    ds_builder.download_and_prepare()
    ds = input_pipeline.create_input_iter(
      ds_builder,
      batch_size,
      self.train_mean,
      self.train_stddev,
      self.center_crop_size,
      self.resize_size,
      self.aspect_ratio_range,
      self.scale_ratio_range,
      train=True,
      cache=False)
    return ds

  def sync_batch_stats(self, model_state):
    """Sync the batch statistics across replicas."""
    # An axis_name is passed to pmap which can then be used by pmean.
    # In this case each device has its own version of the batch statistics and
    # we average them.
    avg_fn = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')
    new_model_state = model_state.copy({
      'batch_stats': avg_fn(model_state['batch_stats'])})
    return new_model_state

  @property
  def param_shapes(self):
    if self._param_shapes is None:
      raise ValueError('This should not happen, workload.init_model_fn() '
                       'should be called before workload.param_shapes!')
    return self._param_shapes

  def initialized(self, key, model):
    input_shape = (1, 224, 224, 3)
    variables = jax.jit(model.init)({'params': key}, jnp.ones(input_shape, model.dtype))
    model_state, params = variables.pop('params')
    return params, model_state

  _InitState = Tuple[spec.ParameterContainer, spec.ModelAuxiliaryState]
  def init_model_fn(
      self,
      rng: spec.RandomState) -> _InitState:
    model_cls = getattr(models, 'ResNet50')
    model = model_cls(num_classes=1000, dtype=jnp.float32)
    self._model = model
    params, model_state = self.initialized(rng, model)
    self._param_shapes = jax.tree_map(
      lambda x: spec.ShapeTuple(x.shape),
      params)
    model_state = jax_utils.replicate(model_state)
    params = jax_utils.replicate(params)
    return params, model_state

    # Keep this separate from the loss function in order to support optimizers
  # that use the logits.
  def output_activation_fn(
      self,
      logits_batch: spec.Tensor,
      loss_type: spec.LossType) -> spec.Tensor:
    """Return the final activations of the model."""
    pass

  @functools.partial(
  jax.pmap,
  axis_name='batch',
  in_axes=(None, 0, 0, 0, None),
  static_broadcasted_argnums=(0,))
  def eval_model_fn(self, params, batch, state, rng):
    logits, _ = self.model_fn(
      params,
      batch,
      state,
      spec.ForwardPassMode.EVAL,
      rng,
      update_batch_norm=False)
    return self.compute_metrics(logits, batch['label'])

  def model_fn(
      self,
      params: spec.ParameterContainer,
      input_batch: spec.Tensor,
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    variables = {'params': params, **model_state}
    train = mode == spec.ForwardPassMode.TRAIN
    if update_batch_norm:
      logits, new_model_state = self._model.apply(
        variables,
        jax.numpy.squeeze(input_batch['image']),
        train=train,
        mutable=['batch_stats'])
      return logits, new_model_state
    else:
      logits = self._model.apply(
        variables,
        jax.numpy.squeeze(input_batch['image']),
        train=train,
        mutable=False)
      return logits, None

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,
      logits_batch: spec.Tensor) -> spec.Tensor:  # differentiable
    """Cross Entropy Loss"""
    one_hot_labels = jax.nn.one_hot(label_batch, num_classes=1000)
    xentropy = optax.softmax_cross_entropy(logits=logits_batch,
                                           labels=one_hot_labels)
    return xentropy

  def compute_metrics(self, logits, labels):
    loss = jnp.mean(self.loss_fn(labels, logits))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
      'loss': loss,
      'accuracy': accuracy,
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics

  def eval_model(
      self,
      params: spec.ParameterContainer,
      model_state: spec.ModelAuxiliaryState,
      rng: spec.RandomState,
      data_dir: str):
    """Run a full evaluation of the model."""
    # sync batch statistics across replicas
    model_state = self.sync_batch_stats(model_state)

    eval_metrics = []
    data_rng, model_rng = prng.split(rng, 2)
    eval_batch_size = 200
    num_batches = self.num_eval_examples // eval_batch_size
    if self._eval_ds is None:
      self._eval_ds = self._build_dataset(
        data_rng, split='test', batch_size=eval_batch_size, data_dir=data_dir)
    eval_iter = iter(self._eval_ds)
    total_accuracy = 0.
    for idx in range(num_batches):
      batch = next(eval_iter)
      synced_metrics = self.eval_model_fn(params, batch, model_state, rng)
      eval_metrics.append(synced_metrics)
      total_accuracy += jnp.mean(synced_metrics['accuracy'])

    eval_metrics = jax.device_get(jax.tree_map(lambda x: x[0], eval_metrics))
    eval_metrics = jax.tree_multimap(lambda *x: np.stack(x), *eval_metrics)
    summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
    return summary

