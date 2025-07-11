import math
from typing import Dict

from algoperf import spec


class BaseLibrispeechWorkload(spec.Workload):
  _num_outputs: int = 1024

  @property
  def target_metric_name(self) -> str:
    """The name of the target metric (useful for scoring/processing code)."""
    return 'wer'

  @property
  def use_post_layer_norm(self) -> bool:
    raise NotImplementedError

  @property
  def use_gelu(self) -> bool:
    raise NotImplementedError

  @property
  def attention_temperature(self) -> float:
    raise NotImplementedError

  def has_reached_validation_target(
    self, eval_result: Dict[str, float]
  ) -> bool:
    return eval_result['validation/wer'] < self.validation_target_value

  @property
  def validation_target_value(self) -> float:
    return 0.085884

  def has_reached_test_target(self, eval_result: Dict[str, float]) -> bool:
    return eval_result['test/wer'] < self.test_target_value

  @property
  def test_target_value(self) -> float:
    return 0.052981

  @property
  def loss_type(self) -> spec.LossType:
    return spec.LossType.CTC_LOSS

  @property
  def num_train_examples(self) -> int:
    return 263840

  @property
  def num_eval_train_examples(self) -> int:
    # Round up from num_validation_examples (which is the default for
    # num_eval_train_examples) to the next multiple of eval_batch_size, so that
    # we don't have to extract the correctly sized subset of the training data.
    rounded_up_multiple = math.ceil(
      self.num_validation_examples / self.eval_batch_size
    )
    return rounded_up_multiple * self.eval_batch_size

  @property
  def num_validation_examples(self) -> int:
    return 5348

  @property
  def num_test_examples(self) -> int:
    return 2472

  @property
  def eval_batch_size(self) -> int:
    return 256

  @property
  def train_mean(self):
    raise NotImplementedError

  @property
  def train_stddev(self):
    raise NotImplementedError

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 58_015  # ~16.1 hours

  @property
  def eval_period_time_sec(self) -> int:
    return 24 * 60

  @property
  def step_hint(self) -> int:
    """Approx. steps the baseline can do in the allowed runtime budget."""
    return 80_000
