"""MNIST workload implemented in Jax."""
import functools
import itertools
from typing import Any, Dict, Optional, Tuple

from flax import jax_utils
from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

from algorithmic_efficiency import data_utils
from algorithmic_efficiency import param_utils
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
    # TODO: choose a random split and match with PyTorch.
    if split == 'eval_train':
      tfds_split = f'train[:{self.num_eval_train_examples}]'
    elif split == 'validation':
      tfds_split = f'train[{self.num_train_examples}:]'
    else:
      tfds_split = f'train[:{self.num_train_examples}]'
    ds = tfds.load(
        'mnist', split=tfds_split, shuffle_files=False, data_dir=data_dir)
    ds = ds.map(lambda x: {
        'inputs': self._normalize(x['image']),
        'targets': x['label'],
    })
    ds = ds.cache()
    is_train = split == 'train'
    if is_train:
      ds = ds.shuffle(16 * batch_size, seed=data_rng[0])
      ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=is_train)
    ds = map(data_utils.shard_numpy_ds, ds)
    return iter(ds)

  # Return whether or not a key in spec.ParameterContainer is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def _build_input_queue(self,
                         data_rng,
                         split: str,
                         data_dir: str,
                         global_batch_size: int) -> Dict[str, Any]:
    ds = self._build_dataset(data_rng, split, data_dir, global_batch_size)
    if split != 'train':
      # Note that this stores the entire eval dataset in memory.
      ds = itertools.cycle(ds)
    return ds

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
    initial_params = self._model.init({'params': rng}, init_val,
                                      train=True)['params']
    self._param_shapes = param_utils.jax_param_shapes(initial_params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
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
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      dropout_rate: Optional[float],
      aux_dropout_rate: Optional[float],
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    """Dropout is unused."""
    del model_state
    del rng
    del dropout_rate
    del aux_dropout_rate
    del update_batch_norm
    train = mode == spec.ForwardPassMode.TRAIN
    logits_batch = self._model.apply(
        {'params': params},
        augmented_and_preprocessed_input_batch['inputs'],
        train=train)
    return logits_batch, None

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(self,
              label_batch: spec.Tensor,
              logits_batch: spec.Tensor,
              label_smoothing: float = 0.0) -> spec.Tensor:  # differentiable
    one_hot_targets = jax.nn.one_hot(label_batch, 10)
    smoothed_targets = optax.smooth_labels(one_hot_targets, label_smoothing)
    return -jnp.sum(smoothed_targets * nn.log_softmax(logits_batch), axis=-1)

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
      rng: spec.RandomState) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    logits, _ = self.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.EVAL,
        rng,
        dropout_rate=None,
        aux_dropout_rate=None,
        update_batch_norm=False)
    accuracy = jnp.sum(jnp.argmax(logits, axis=-1) == batch['targets'])
    loss = jnp.sum(self.loss_fn(batch['targets'], logits))
    metrics = {'accuracy': accuracy, 'loss': loss}
    metrics = lax.psum(metrics, axis_name='batch')
    return metrics
