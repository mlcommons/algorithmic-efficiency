"""MNIST workload implemented in Jax."""
import functools
from typing import Tuple

from flax import jax_utils
from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.mnist.workload import BaseMnistWorkload


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


def _param_types(param_tree):
  param_types_dict = {}
  for name, value in param_tree.items():
    if isinstance(value, dict):
      param_types_dict[name] = _param_types(value)
    else:
      if 'bias' in name:
        param_types_dict[name] = spec.ParameterType.BIAS
      else:
        param_types_dict[name] = spec.ParameterType.WEIGHT
  return param_types_dict


class MnistWorkload(BaseMnistWorkload):

  def __init__(self):
    super().__init__()
    self._model = _Model()

  def _normalize(self, image):
    return (tf.cast(image, tf.float32) - self.train_mean) / self.train_stddev

  def _build_dataset(self,
                     data_rng: jax.random.PRNGKey,
                     split: str,
                     data_dir: str,
                     batch_size):
    if split == 'eval_train':
      tfds_split = 'train[:50000]'
    elif split == 'validation':
      tfds_split = 'train[50000:]'
    else:
      tfds_split = split
    ds = tfds.load(
        'mnist', split=tfds_split, shuffle_files=False, data_dir=data_dir)
    ds = ds.cache()
    ds = ds.map(lambda x: (self._normalize(x['image']), x['label']))
    if split == 'train':
      ds = ds.shuffle(1024, seed=data_rng[0])
      ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.batch(jax.local_device_count())
    return tfds.as_numpy(ds)

  @property
  def param_shapes(self):
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    return self._param_shapes

  @property
  def model_params_types(self):
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    if self._param_types is None:
      self._param_types = _param_types(self._param_shapes.unfreeze())
    return self._param_types

  # Return whether or not a key in spec.ParameterContainer is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def build_input_queue(self,
                        data_rng,
                        split: str,
                        data_dir: str,
                        global_batch_size: int):
    ds = self._build_dataset(data_rng, split, data_dir, global_batch_size)
    for images, labels in iter(ds):
      yield images, labels, None

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
    initial_params = self._model.init({'params': rng}, init_val,
                                      train=True)['params']
    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      initial_params)
    return jax_utils.replicate(initial_params), None

  # Keep this separate from the loss function in order to support optimizers
  # that use the logits.
  def output_activation_fn(self,
                           logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:
    if loss_type == spec.LossType.SOFTMAX_CROSS_ENTROPY:
      return jax.nn.softmax(logits_batch, axis=-1)
    if loss_type == spec.LossType.SIGMOID_CROSS_ENTROPY:
      return jax.nn.sigmoid(logits_batch)
    if loss_type == spec.LossType.MEAN_SQUARED_ERROR:
      return logits_batch

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: spec.Tensor,
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del rng
    del update_batch_norm
    train = mode == spec.ForwardPassMode.TRAIN
    logits_batch = self._model.apply({'params': params},
                                     augmented_and_preprocessed_input_batch,
                                     train=train)
    return logits_batch, None

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(self, label_batch: spec.Tensor,
              logits_batch: spec.Tensor) -> spec.Tensor:  # differentiable
    one_hot_targets = jax.nn.one_hot(label_batch, 10)
    return -jnp.sum(one_hot_targets * nn.log_softmax(logits_batch), axis=-1)

  @functools.partial(
      jax.pmap,
      axis_name='batch',
      in_axes=(None, 0, 0, 0, 0, None),
      static_broadcasted_argnums=(0,))
  def _eval_model(
      self,
      params: spec.ParameterContainer,
      images: spec.Tensor,
      labels: spec.Tensor,
      model_state: spec.ModelAuxiliaryState,
      rng: spec.RandomState) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    logits, _ = self.model_fn(
        params,
        images,
        model_state,
        spec.ForwardPassMode.EVAL,
        rng,
        update_batch_norm=False)
    accuracy = jnp.sum(jnp.argmax(logits, axis=-1) == labels)
    loss = jnp.sum(self.loss_fn(labels, logits))
    num_data = len(logits)
    metrics = {'accuracy': accuracy, 'loss': loss, 'num_data': num_data}
    metrics = lax.psum(metrics, axis_name='batch')
    return metrics
