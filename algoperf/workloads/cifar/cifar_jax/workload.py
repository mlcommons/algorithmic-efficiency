"""CIFAR workload implemented in Jax."""

import functools
from typing import Any, Dict, Iterator, Optional, Tuple

from flax import linen as nn
from flax.core import pop
import jax
from jax import lax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds

from algoperf import param_utils
from algoperf import sharding_utils
from algoperf import spec
from algoperf.workloads.cifar.cifar_jax import models
from algoperf.workloads.cifar.cifar_jax.input_pipeline import create_input_iter
from algoperf.workloads.cifar.workload import BaseCifarWorkload


class CifarWorkload(BaseCifarWorkload):

  def _build_cifar_dataset(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      batch_size: int,
      cache: Optional[bool] = None,
      repeat_final_dataset: Optional[bool] = None,
  ) -> Iterator[Dict[str, spec.Tensor]]:
    data_dir = data_dir + "/cifar10"
    ds_builder = tfds.builder("cifar10:3.0.2", data_dir=data_dir)
    train = split == "train"
    assert self.num_train_examples + self.num_validation_examples == 50000
    if split in ["train", "eval_train"]:
      split = f"train[:{self.num_train_examples}]"
    elif split == "validation":
      split = f"train[{self.num_train_examples}:]"
    ds = create_input_iter(
        split,
        ds_builder,
        data_rng,
        batch_size,
        self.train_mean,
        self.train_stddev,
        self.crop_size,
        self.padding_size,
        train=train,
        cache=not train if cache is None else cache,
        repeat_final_dataset=repeat_final_dataset,
    )
    return ds

  def _build_input_queue(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      global_batch_size: int,
      cache: Optional[bool] = None,
      repeat_final_dataset: Optional[bool] = None,
      num_batches: Optional[int] = None,
  ) -> Iterator[Dict[str, spec.Tensor]]:
    del num_batches
    return self._build_cifar_dataset(data_rng,
                                     split,
                                     data_dir,
                                     global_batch_size,
                                     cache,
                                     repeat_final_dataset)

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None,
  ) -> spec.ModelInitState:
    """Dropout is unused."""
    del dropout_rate
    del aux_dropout_rate
    model_cls = getattr(models, "ResNet18")
    model = model_cls(num_classes=self._num_classes, dtype=jnp.float32)
    self._model = model
    input_shape = (1, 32, 32, 3)
    variables = jax.jit(model.init)({"params": rng},
                                    jnp.ones(input_shape, model.dtype))
    model_state, params = pop(variables, 'params')
    self._param_shapes = param_utils.jax_param_shapes(params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
    return params, model_state

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key == "Dense_0"

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool,
      use_running_average_bn: Optional[bool] = None
  ) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del mode
    del rng
    variables = {"params": params, **model_state}
    if update_batch_norm:
      logits, new_model_state = self._model.apply(
          variables,
          augmented_and_preprocessed_input_batch["inputs"],
          update_batch_norm=update_batch_norm,
          mutable=['batch_stats'],
          use_running_average_bn=use_running_average_bn)
      return logits, new_model_state
    else:
      logits = self._model.apply(
          variables,
          augmented_and_preprocessed_input_batch["inputs"],
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
      label_smoothing: float = 0.0,
  ) -> Dict[str, spec.Tensor]:  # differentiable
    """Evaluate the (masked) loss function at (label_batch, logits_batch).

        Return {'summed': scalar summed loss,
        'n_valid_examples': scalar number of
        valid examples in batch, 'per_example': 1-d array of per-example losses}
        (not synced across devices).
        """
    one_hot_targets = jax.nn.one_hot(label_batch, self._num_classes)
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
        "summed": summed_loss,
        "n_valid_examples": n_valid_examples,
        "per_example": per_example_losses,
    }

  def _compute_metrics(self,
                       logits: spec.Tensor,
                       labels: spec.Tensor,
                       weights: spec.Tensor) -> Dict[str, spec.Tensor]:
    summed_loss = self.loss_fn(labels, logits, weights)["summed"]
    # Number of correct predictions.
    accuracy = jnp.sum((jnp.argmax(logits, -1) == labels) * weights)
    return jnp.array(summed_loss), jnp.array(accuracy)

  def _eval_model(
      self,
      params: spec.ParameterContainer,
      batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      rng: spec.RandomState,
  ) -> Dict[spec.Tensor, spec.ModelAuxiliaryState]:
    """Return the mean accuracy and loss as a dict."""

    @functools.partial(
        jax.jit,
        in_shardings=(
            sharding_utils.get_replicated_sharding(),  # params
            sharding_utils.get_naive_sharding_spec(),  # batch
            sharding_utils.get_replicated_sharding(),  # model_state
            sharding_utils.get_naive_sharding_spec(),  # rng
        ),
    )
    def _per_device_eval_model(
        params: spec.ParameterContainer,
        batch: Dict[str, spec.Tensor],
        model_state: spec.ModelAuxiliaryState,
        rng: spec.RandomState,
    ) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
      logits, _ = self.model_fn(
          params,
          batch,
          model_state,
          spec.ForwardPassMode.EVAL,
          rng,
          update_batch_norm=False,
      )
      weights = batch.get("weights")
      if weights is None:
        weights = jnp.ones(len(logits))
      return self._compute_metrics(logits, batch["targets"], weights)

    losses, accuracies = _per_device_eval_model(params, batch, model_state, rng)
    metrics = {
        "loss":
            jnp.mean(losses, axis=0) if losses.ndim > 0 else losses,
        "accuracy":
            (jnp.mean(accuracies, axis=0) if accuracies.ndim > 0 else accuracies
            ),
    }
    return metrics

  def _normalize_eval_metrics(
      self, num_examples: int, total_metrics: Dict[str,
                                                   Any]) -> Dict[str, float]:
    """Normalize eval metrics."""
    return jax.tree_map(lambda x: x / num_examples, total_metrics)
