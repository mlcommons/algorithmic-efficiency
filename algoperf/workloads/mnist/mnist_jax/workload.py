"""MNIST workload implemented in Jax."""

import functools
from typing import Any, Dict, Optional, Tuple

from flax import jax_utils
from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import optax

from algoperf import param_utils
from algoperf import sharding_utils
from algoperf import spec
from algoperf.workloads.mnist.workload import BaseMnistWorkload


class _Model(nn.Module):

  @nn.compact
  def __call__(self, x: spec.Tensor, train: bool) -> spec.Tensor:
    del train
    input_size = 28 * 28
    num_hidden = 128
    num_classes = 10
    x = x.reshape((x.shape[0], input_size))
    x = nn.Dense(features=num_hidden, use_bias=True)(x)
    x = nn.sigmoid(x)
    x = nn.Dense(features=num_classes, use_bias=True)(x)
    return x


class MnistWorkload(BaseMnistWorkload):

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
    return initial_params, None

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key == 'Dense_1'

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
      label_batch: spec.Tensor,  # Dense or one-hot labels.
      logits_batch: spec.Tensor,
      mask_batch: Optional[spec.Tensor] = None,
      label_smoothing: float = 0.0) -> Dict[str, spec.Tensor]:  # differentiable
    """Evaluate the (masked) loss function at (label_batch, logits_batch).

    Return {'summed': scalar summed loss, 'n_valid_examples': scalar number of
    valid examples in batch, 'per_example': 1-d array of per-example losses}
    (not synced across devices).
    """
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
    return {
        'summed': summed_loss,
        'n_valid_examples': n_valid_examples,
        'per_example': per_example_losses,
    }

  @functools.partial(
      jax.jit,
      in_shardings=(
          sharding_utils.get_replicated_sharding(),  # params
          sharding_utils.get_naive_sharding_spec(),  # batch
          sharding_utils.get_replicated_sharding(),  # model_state
          sharding_utils.get_naive_sharding_spec(),  # rng
      ),
      static_argnums=(0,))
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
    summed_loss = self.loss_fn(batch['targets'], logits, weights)['summed']
    metrics = {'accuracy': accuracy, 'loss': summed_loss}
    return metrics

  def _normalize_eval_metrics(
      self, num_examples: int, total_metrics: Dict[str,
                                                   Any]) -> Dict[str, float]:
    """Normalize eval metrics."""
    total_metrics = {
        'accuracy': total_metrics['accuracy'].item() / num_examples,
        'loss': total_metrics['loss'].item() / num_examples
    }
    return total_metrics
