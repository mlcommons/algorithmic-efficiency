"""OGBG workload parent class."""

import abc
import itertools
import math
from typing import Any, Dict, Optional, Tuple

import jax

from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.ogbg import input_pipeline
from algorithmic_efficiency.workloads.ogbg import metrics


class BaseOgbgWorkload(spec.Workload):

  _num_outputs: int = 128

  @property
  def target_metric_name(self) -> str:
    """The name of the target metric (useful for scoring/processing code)."""
    return 'mean_average_precision'

  @property 
  def activation_fn_name(self) -> str:
    """Name of the activation function to use. One of 'relu', 'gelu', 'silu'."""
    return 'relu'

  @property
  def hidden_dims(self) -> Tuple[int]:
    return (256,)

  @property
  def latent_dim(self) -> int:
    return 256

  @property
  def num_message_passing_steps(self) -> int:
    return 5

  def has_reached_validation_target(self, eval_result: float) -> bool:
    return eval_result[
        'validation/mean_average_precision'] > self.validation_target_value

  @property
  def validation_target_value(self) -> float:
    return 0.28098

  def has_reached_test_target(self, eval_result: float) -> bool:
    return eval_result['test/mean_average_precision'] > self.test_target_value

  @property
  def test_target_value(self) -> float:
    return 0.268729

  @property
  def loss_type(self) -> spec.LossType:
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self) -> int:
    return 350343

  @property
  def num_eval_train_examples(self) -> int:
    return 43793

  @property
  def num_validation_examples(self) -> int:
    return 43793

  @property
  def num_test_examples(self) -> int:
    return 43793

  @property
  def eval_batch_size(self) -> int:
    return 32768

  @property
  def train_mean(self):
    raise NotImplementedError

  @property
  def train_stddev(self):
    raise NotImplementedError

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 18_477  # ~5 hours

  @property
  def eval_period_time_sec(self) -> int:
    return 4 * 60

  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int):
    dataset_iter = input_pipeline.get_dataset_iter(split,
                                                   data_rng,
                                                   data_dir,
                                                   global_batch_size)
    if split != 'train':
      # Note that this stores the entire val dataset in memory.
      dataset_iter = itertools.cycle(dataset_iter)
    return dataset_iter

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
    per_example_losses = self._binary_cross_entropy_with_mask(
        labels=label_batch,
        logits=logits_batch,
        mask=mask_batch,
        label_smoothing=label_smoothing)
    if mask_batch is not None:
      n_valid_examples = mask_batch.sum()
    else:
      n_valid_examples = len(per_example_losses)
    summed_loss = per_example_losses.sum()
    return {
        'summed': summed_loss,
        'n_valid_examples': n_valid_examples,
        'per_example': per_example_losses,
    }

  @property
  def step_hint(self) -> int:
    """Max num steps the baseline algo was given to reach the target."""
    return 80_000

  @abc.abstractmethod
  def _normalize_eval_metrics(
      self, num_examples: int, total_metrics: Dict[str,
                                                   Any]) -> Dict[str, float]:
    """Normalize eval metrics."""

  def _eval_batch(self,
                  params: spec.ParameterContainer,
                  batch: Dict[str, spec.Tensor],
                  model_state: spec.ModelAuxiliaryState,
                  rng: spec.RandomState) -> metrics.EvalMetrics:
    logits, _ = self.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.EVAL,
        rng,
        update_batch_norm=False)
    return self._eval_metric(batch['targets'], logits, batch['weights'])

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str,
                           global_step: int = 0) -> Dict[str, float]:
    """Run a full evaluation of the model."""
    del global_step
    data_rng, model_rng = prng.split(rng, 2)
    if split not in self._eval_iters:
      self._eval_iters[split] = self._build_input_queue(
          data_rng, split, data_dir, global_batch_size=global_batch_size)

    total_metrics = None
    num_eval_steps = int(math.ceil(float(num_examples) / global_batch_size))
    # Loop over graph batches in eval dataset.
    for _ in range(num_eval_steps):
      batch = next(self._eval_iters[split])
      batch_metrics = self._eval_batch(params, batch, model_state, model_rng)
      total_metrics = (
          batch_metrics
          if total_metrics is None else total_metrics.merge(batch_metrics))
    if total_metrics is None:
      return {}
    return self._normalize_eval_metrics(num_examples, total_metrics)
