"""LM workload parent class."""

import abc
import math
import os
from typing import Dict, Optional

from absl import flags
import jax
import torch.distributed as dist

from algoperf import spec
from algoperf.workloads.lm import input_pipeline

FLAGS = flags.FLAGS

USE_PYTORCH_DDP = 'LOCAL_RANK' in os.environ


class BaseLmWorkload(spec.Workload):
  """LM workload."""

  _vocab_size: int = 50257
  _seq_len: int = 512

  def __init__(self) -> None:
    pass

  @property
  def target_metric_name(self) -> str:
    """The name of the target metric (useful for scoring/processing code)."""
    return 'ppl'

  def has_reached_validation_target(self, eval_result: float) -> bool:
    return eval_result['validation/ppl'] > self.validation_target_value

  @property
  def validation_target_value(self) -> float:
    pass

  def has_reached_test_target(self, eval_result: float) -> bool:
    return eval_result['test/ppl'] > self.test_target_value

  @property
  def test_target_value(self) -> float:
    pass

  @property
  def loss_type(self) -> spec.LossType:
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self) -> int:
    pass

  @property
  def num_eval_train_examples(self) -> int:
    pass

  @property
  def num_validation_examples(self) -> int:
    pass

  @property
  def num_test_examples(self) -> int:
    pass

  @property
  def eval_batch_size(self) -> int:
    pass

  @property
  def train_mean(self):
    raise NotImplementedError

  @property
  def train_stddev(self):
    raise NotImplementedError

  @property
  def max_allowed_runtime_sec(self) -> int:
    pass

  @property
  def eval_period_time_sec(self) -> int:
    pass

  @property
  def step_hint(self) -> int:
    """Approx. steps the baseline can do in the allowed runtime budget."""
    pass

  @property
  def pre_ln(self) -> bool:
    return True

  @property
  def attention_temp(self) -> float:
    return 1.0

  @property
  def activation(self) -> str:
    return 'silu'

  @property
  def glu(self) -> bool:
    return True

  @abc.abstractmethod
  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         num_batches: Optional[int] = None,
                         repeat_final_dataset: bool = False):
    """Build an input queue for the given split."""

  @abc.abstractmethod
  def _eval_batch(self,
                  params: spec.ParameterContainer,
                  batch: Dict[str, spec.Tensor],
                  model_state: spec.ModelAuxiliaryState,
                  rng: spec.RandomState) -> spec.Tensor:
    """Evaluate the model on a single batch."""

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

    for _ in range(num_batches):
      eval_batch = next(self._eval_iters[split])
      loss += self._eval_batch(params, eval_batch)
    if USE_PYTORCH_DDP:
      dist.all_reduce(loss)
    mean_loss = loss.item() / num_examples
    return {'loss': mean_loss}

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
    pass
