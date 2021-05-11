"""MNIST workload implemented in Jax."""

import struct
import time
from typing import Tuple
from workloads.mnist.workload import Mnist

from flax import linen as nn

import jax
import jax.numpy as jnp
import spec
import tensorflow as tf
import tensorflow_datasets as tfds


class _Model(nn.Module):

  @nn.compact
  def __call__(self, x: spec.Tensor, train: bool):
    del train
    input_size = 28 * 28
    num_hidden = 128
    num_classes = 10
    x = x.reshape((x.shape[0], input_size))  # Flatten.
    x = nn.Dense(features=num_hidden, use_bias=True)(x)
    x = nn.sigmoid(x)
    x = nn.Dense(features=num_classes, use_bias=True)(x)
    x = nn.log_softmax(x)
    return x


class MnistWorkload(Mnist):

  def __init__(self):
    self._eval_ds = None
    self._param_shapes = None
    self._model = _Model()

  def _normalize(self, image):
    return (tf.cast(image, tf.float32) - self.train_mean) / self.train_stddev

  def _build_dataset(self,
      data_rng: jax.random.PRNGKey,
      split: str,
      data_dir: str,
      batch_size):
    ds = tfds.load('mnist', split=split, try_gcs=True)
    ds = ds.cache()
    ds = ds.map(lambda x: (self._normalize(x['image']), x['label']))
    if split == 'train':
      ds = ds.shuffle(1024, seed=data_rng[0])
      ds = ds.repeat()
    ds = ds.batch(batch_size)
    return tfds.as_numpy(ds)

  def build_input_queue(
      self,
      data_rng: jax.random.PRNGKey,
      split: str,
      data_dir: str,
      batch_size: int):
    return iter(self._build_dataset(data_rng, split, data_dir, batch_size))

  @property
  def param_shapes(self):
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    return self._param_shapes

  def model_params_types(self):
    pass

  # Return whether or not a key in spec.ParameterTree is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def preprocess_for_train(
      self,
      selected_raw_input_batch: spec.Tensor,
      selected_label_batch: spec.Tensor,
      train_mean: spec.Tensor,
      train_stddev: spec.Tensor,
      rng: spec.RandomState) -> spec.Tensor:
    del rng
    return selected_raw_input_batch, selected_label_batch

  def preprocess_for_eval(
      self,
      raw_input_batch: spec.Tensor,
      raw_label_batch: spec.Tensor,
      train_mean: spec.Tensor,
      train_stddev: spec.Tensor) -> spec.Tensor:
    del train_mean
    del train_stddev
    return raw_input_batch, raw_label_batch

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
    initial_params = self._model.init(rng, init_val, train=True)['params']
    self._param_shapes = jax.tree_map(
        lambda x: spec.ShapeTuple(x.shape), initial_params)
    return initial_params, None

  def model_fn(
      self,
      params: spec.ParameterTree,
      augmented_and_preprocessed_input_batch: spec.Tensor,
      model_state: spec.ModelAuxillaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxillaryState]:
    del model_state
    del rng
    del update_batch_norm
    train = mode == spec.ForwardPassMode.TRAIN
    logits_batch = self._model.apply(
        {'params': params}, augmented_and_preprocessed_input_batch, train=train)
    return logits_batch, None

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,
      logits_batch: spec.Tensor) -> spec.Tensor:  # differentiable
    one_hot_targets = jax.nn.one_hot(label_batch, 10)
    return -jnp.sum(one_hot_targets * nn.log_softmax(logits_batch), axis=-1)

  def _eval_metric(self, logits, labels):
    """Return the mean accuracy and loss as a dict."""
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    loss = jnp.mean(self.loss_fn(labels, logits))
    return {'accuracy': accuracy, 'loss': loss}
