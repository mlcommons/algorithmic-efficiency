"""WMT workload parent class."""

import abc
import math
import os
from typing import Any, Dict, Optional, Tuple

import jax
import numpy as np
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.wmt import input_pipeline
from algorithmic_efficiency.workloads.wmt.wmt_jax import decode

VOCAB_PATH = './wmt_256/sentencepiece_model'
WORKDIR = './wmt_256'
USE_PYTORCH_DDP = 'LOCAL_RANK' in os.environ


class BaseWmtWorkload(spec.Workload):
  """A WMT workload."""

  _vocab_size: int = 32000

  def __init__(self) -> None:
    super().__init__()
    self._tokenizer = None

  @property
  def target_metric_name(self) -> str:
    """The name of the target metric (useful for scoring/processing code)."""
    return 'bleu'

  def has_reached_validation_target(self, eval_result: float) -> bool:
    return eval_result['validation/bleu'] > self.validation_target_value

  @property
  def validation_target_value(self) -> float:
    return 30.8491

  def has_reached_test_target(self, eval_result: float) -> bool:
    return eval_result['test/bleu'] > self.test_target_value

  @property
  def test_target_value(self) -> float:
    return 30.7219

  @property
  def loss_type(self) -> spec.LossType:
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self) -> int:
    # wmt17_translate/de-en 'train' split size
    return 5906184

  @property
  def num_eval_train_examples(self) -> int:
    # Round up from num_validation_examples (which is the default for
    # num_eval_train_examples) to the next multiple of eval_batch_size, so that
    # we don't have to extract the correctly sized subset of the training data.
    rounded_up_multiple = math.ceil(self.num_validation_examples /
                                    self.eval_batch_size)
    return rounded_up_multiple * self.eval_batch_size

  @property
  def num_validation_examples(self) -> int:
    # wmt14_translate/de-en 'validation' split size.
    return 3000

  @property
  def num_test_examples(self) -> int:
    # wmt14_translate/de-en 'test' split size.
    return 3003

  @property
  def eval_batch_size(self) -> int:
    return 128

  @property
  def train_mean(self):
    raise NotImplementedError

  @property
  def train_stddev(self):
    raise NotImplementedError

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 48_151  # ~13.5 hours

  @property
  def eval_period_time_sec(self) -> int:
    return 14 * 60

  @property
  def step_hint(self) -> int:
    """Max num steps the baseline algo was given to reach the target."""
    return 133_333

  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         num_batches: Optional[int] = None,
                         repeat_final_dataset: bool = False):
    is_training = split == 'train'
    ds, self._tokenizer = input_pipeline.get_wmt_dataset(
        data_rng,
        split,
        data_dir,
        is_training=is_training,
        vocab_size=self._vocab_size,
        global_batch_size=global_batch_size,
        num_batches=num_batches,
        repeat_final_dataset=repeat_final_dataset)

    # Separate function is necessary because the code above has to be executed
    # when _build_input_queue is called (not when next() is first called on it).
    def _input_queue_generator():
      for batch in iter(ds):
        weights = batch.get('weights')
        updated_weights = np.where(batch['targets'] > 0, 1, 0)
        if weights is not None:
          updated_weights = np.logical_and(weights, updated_weights)
        batch['weights'] = updated_weights
        yield batch

    return _input_queue_generator()

  @abc.abstractmethod
  def _normalize_eval_metrics(
      self, num_examples: int, total_metrics: Dict[str,
                                                   Any]) -> Dict[str, float]:
    """Normalize eval metrics."""

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
    del model_state
    del global_step
    num_batches = int(math.ceil(num_examples / global_batch_size))
    if split not in self._eval_iters:
      # These iterators will repeat indefinitely.
      self._eval_iters[split] = self._build_input_queue(
          rng,
          split,
          data_dir,
          global_batch_size,
          num_batches,
          repeat_final_dataset=True)

    eval_metrics = {}
    for _ in range(num_batches):
      eval_batch = next(self._eval_iters[split])
      metrics = self.eval_step(params, eval_batch)
      for metric_name, metric_value in metrics.items():
        if metric_name not in eval_metrics:
          eval_metrics[metric_name] = 0.0
        eval_metrics[metric_name] += metric_value
    eval_results = self._normalize_eval_metrics(num_examples, eval_metrics)

    eval_results['bleu'] = self.translate_and_calculate_bleu(
        params=params,
        ds_iter=self._eval_iters[split],
        num_batches=num_batches,
        max_predict_length=256)

    return eval_results

  def compute_weighted_accuracy(
      self, logits: spec.Tensor, targets: spec.Tensor,
      weights: spec.Tensor) -> Tuple[spec.Tensor, spec.Tensor]:
    """Compute weighted accuracy for log probs and targets.

    Args:
      logits: [batch, length, num_classes] float array.
      targets: categorical targets [batch, length] int array.
      weights: array of shape [batch, length]

    Returns:
      Tuple of scalar summed accuracy and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
      raise ValueError(f'Incorrect shapes. Got shape {logits.shape} logits and '
                       f'{targets.shape} targets.')
    accuracy = (logits.argmax(-1) == targets) * weights
    normalizing_factor = weights.sum()
    return accuracy.sum(), normalizing_factor

  def _decode_tokens(self, toks: spec.Tensor) -> spec.Tensor:
    if isinstance(toks, torch.Tensor):
      toks = toks.cpu().numpy()
    valid_toks = toks[:np.argmax(toks == decode.EOS_ID) + 1].astype(np.int32)
    return self._tokenizer.detokenize(valid_toks).numpy().decode('utf-8')

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
    return self.compute_weighted_cross_entropy(
        logits_batch,
        label_batch,
        weights=mask_batch,
        label_smoothing=label_smoothing)
