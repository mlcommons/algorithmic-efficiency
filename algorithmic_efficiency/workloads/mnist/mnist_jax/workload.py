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


class MnistWorkload(BaseMnistWorkload):

  def _normalize(self, image):
    return (tf.cast(image, tf.float32) - self.train_mean) / self.train_stddev

  def _build_dataset(self,
                     data_rng: jax.random.PRNGKey,
                     split: str,
                     data_dir: str,
                     global_batch_size: int):
    shuffle = split in ['train', 'eval_train']
    if shuffle:
      tfds_split = f'train[:{self.num_train_examples}]'
    elif split == 'validation':
      tfds_split = f'train[{self.num_train_examples}:]'
    else:
      tfds_split = 'test'
    ds = tfds.load(
        'mnist:3.0.1', split=tfds_split, shuffle_files=False, data_dir=data_dir)
    ds = ds.map(lambda x: {
        'inputs': self._normalize(x['image']),
        'targets': x['label'],
    })
    ds = ds.cache()
    is_train = split == 'train'
    if shuffle:
      ds = ds.shuffle(16 * global_batch_size, seed=data_rng[0])
      if is_train:
        ds = ds.repeat()
    ds = ds.batch(global_batch_size, drop_remainder=is_train)
    ds = map(
        functools.partial(
            data_utils.shard_and_maybe_pad_np,
            global_batch_size=global_batch_size),
        ds)
    return iter(ds)

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key == 'Dense_1'

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

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """Dropout is unused."""
    del dropout_rate
    del aux_dropout_rate
    init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
    self._model = _Model()
    initial_params = self._model.init({'params': rng}, init_val,
                                      train=True)['params']
    self._param_shapes = param_utils.jax_param_shapes(initial_params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
    return jax_utils.replicate(initial_params), None

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
    logits_batch = self._model.apply(
        {'params': params},
        augmented_and_preprocessed_input_batch['inputs'],
        train=train)
    return logits_batch, None

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,
      logits_batch: spec.Tensor,
      mask_batch: Optional[spec.Tensor] = None,
      label_smoothing: float = 0.0
  ) -> Tuple[spec.Tensor, spec.Tensor]:  # differentiable
    """Return (correct scalar average loss, 1-d array of per-example losses)."""
    one_hot_targets = jax.nn.one_hot(label_batch, 10)
    smoothed_targets = optax.smooth_labels(one_hot_targets, label_smoothing)
    per_example_losses = -jnp.sum(
        smoothed_targets * nn.log_softmax(logits_batch), axis=-1)
    # mask_batch is assumed to be shape [batch].
    if mask_batch is not None:
      per_example_losses *= mask_batch
      n_valid_examples = mask_batch.sum()
    else:
      n_valid_examples = len(per_example_losses)
    summed_loss = per_example_losses.sum()
    return summed_loss / n_valid_examples, per_example_losses

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
        update_batch_norm=False)
    weights = batch.get('weights')
    if weights is None:
      weights = jnp.ones(len(logits))
    accuracy = jnp.sum(
        (jnp.argmax(logits, axis=-1) == batch['targets']) * weights)
    _, per_example_losses = self.loss_fn(batch['targets'], logits, weights)
    loss = jnp.sum(per_example_losses)
    metrics = {'accuracy': accuracy, 'loss': loss}
    metrics = lax.psum(metrics, axis_name='batch')
    return metrics
