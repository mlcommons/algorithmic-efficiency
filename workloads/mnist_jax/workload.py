"""MNIST workload implemented in Jax."""

import struct
import time
from typing import Tuple

from flax import linen as nn

import jax
import jax.numpy as jnp
import spec
import tensorflow_datasets as tfds


class _Model(nn.Module):

  @nn.compact
  def __call__(self, x: spec.Tensor, train: bool):
    del train
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    x = nn.log_softmax(x)
    return x


class MnistWorkload(spec.Workload):

  def __init__(self):
    self._eval_ds = None

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result > 0.9

  def _build_dataset(self,
      data_rng: jax.random.PRNGKey,
      split: str,
      batch_size):
    ds = tfds.load('mnist', split=split, try_gcs=True)
    ds = ds.cache()
    if split == 'train':
      ds = ds.shuffle(1024, seed=data_rng[0])
      ds = ds.repeat()
    ds = ds.batch(batch_size)
    return tfds.as_numpy(ds)

  def build_input_queue(
      self,
      data_rng: jax.random.PRNGKey,
      split: str,
      batch_size: int):
    return iter(self._build_dataset(data_rng, split, batch_size))

  @property
  def param_shapes(self):
    init_params, _ = self.init_model_fn(jax.random.PRNGKey(0))
    return jax.tree_map(lambda x: spec.ShapeTuple(x.shape), init_params)

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def train_mean(self):
    return 0.0

  @property
  def train_stddev(self):
    return 1.0

  def model_params_types(self):
    pass

  @property
  def max_allowed_runtime_sec(self):
    return 60

  @property
  def eval_period_time_sec(self):
    return 10

  # Return whether or not a key in spec.ParameterTree is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def preprocess_for_train(
      self,
      selected_raw_input_batch: spec.Tensor,
      selected_label_batch: spec.Tensor,
      rng: spec.RandomState) -> spec.Tensor:
    del rng
    return self.preprocess_for_eval(
        selected_raw_input_batch, selected_label_batch, None, None)

  def preprocess_for_eval(
      self,
      raw_input_batch: spec.Tensor,
      raw_label_batch: spec.Tensor,
      train_mean: spec.Tensor,
      train_stddev: spec.Tensor) -> spec.Tensor:
    del train_mean
    del train_stddev
    return (raw_input_batch, raw_label_batch)

  _InitState = Tuple[spec.ParameterTree, spec.ModelAuxillaryState]
  def init_model_fn(self, rng: spec.RandomState) -> _InitState:
    init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
    initial_params = _Model().init(rng, init_val, train=True)['params']
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
    logits_batch = _Model().apply(
        {'params': params}, augmented_and_preprocessed_input_batch, train=train)
    return logits_batch, None

  # LossFn = Callable[Tuple[spec.Tensor, spec.Tensor], spec.Tensor]
  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,
      logits_batch: spec.Tensor,
      loss_type: spec.LossType) -> spec.Tensor:  # differentiable
    del loss_type
    one_hot_targets = jax.nn.one_hot(label_batch, 10)
    return -jnp.sum(one_hot_targets * nn.log_softmax(logits_batch), axis=-1)

  def eval_model(
      self,
      params: spec.ParameterTree,
      model_state: spec.ModelAuxillaryState,
      rng: spec.RandomState):
    """Run a full evaluation of the model."""
    data_rng, model_rng = jax.random.split(rng, 2)
    eval_batch_size = 2000
    num_batches = 10000 // eval_batch_size
    if self._eval_ds is None:
      self._eval_ds = self._build_dataset(
          data_rng, split='test', batch_size=eval_batch_size)
    eval_iter = iter(self._eval_ds)
    total_loss = 0.
    total_accuracy = 0.
    for x in eval_iter:
      images = x['image']
      labels = x['label']
      logits, _ = self.model_fn(
          params,
          images,
          model_state,
          spec.ForwardPassMode.EVAL,
          model_rng,
          update_batch_norm=False)
      # TODO(znado): add additional eval metrics?
      # total_loss += self.loss_fn(labels, logits, self.loss_type)
      total_accuracy += jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return float(total_accuracy / num_batches)
