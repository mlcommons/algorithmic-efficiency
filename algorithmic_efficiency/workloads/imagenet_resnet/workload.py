"""ImageNet workload parent class."""

import math
from typing import Dict, Iterator, Optional, Tuple

from algorithmic_efficiency import spec


class BaseImagenetResNetWorkload(spec.Workload):

  _num_classes: int = 1000

  def has_reached_goal(self, eval_result: Dict[str, float]) -> bool:
    return eval_result['validation/accuracy'] > self.target_value

  @property
  def target_value(self) -> float:
    return 0.77185  # TODO(namanagarwal): This will edited again soon.

  @property
  def loss_type(self) -> spec.LossType:
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self) -> int:
    return 1281167

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
    return 50000

  @property
  def num_test_examples(self) -> int:
    return 10000  # ImageNet-v2.

  @property
  def eval_batch_size(self) -> int:
    return 1024

  @property
  def train_mean(self) -> Tuple[float, float, float]:
    return (0.485 * 255, 0.456 * 255, 0.406 * 255)

  @property
  def train_stddev(self) -> Tuple[float, float, float]:
    return (0.229 * 255, 0.224 * 255, 0.225 * 255)

  # Data augmentation settings.

  @property
  def scale_ratio_range(self) -> Tuple[float, float]:
    return (0.08, 1.0)

  @property
  def aspect_ratio_range(self) -> Tuple[float, float]:
    return (0.75, 4.0 / 3.0)

  @property
  def center_crop_size(self) -> int:
    return 224

  @property
  def resize_size(self) -> int:
    return 256

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 520  # 31 hours.

  @property
  def eval_period_time_sec(self) -> int:
    return 510  # 8.5 minutes.

  def _build_dataset(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      global_batch_size: int,
      cache: Optional[bool] = None,
      repeat_final_dataset: Optional[bool] = None,
      use_mixup: bool = False,
      use_randaug: bool = False) -> Iterator[Dict[str, spec.Tensor]]:
    raise NotImplementedError

  def _build_input_queue(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      global_batch_size: int,
      cache: Optional[bool] = None,
      repeat_final_dataset: Optional[bool] = None,
      num_batches: Optional[int] = None) -> Iterator[Dict[str, spec.Tensor]]:
    del num_batches
    if split == 'test':
      if not cache:
        raise ValueError('cache must be True for split=test.')
      if not repeat_final_dataset:
        raise ValueError('repeat_final_dataset must be True for split=test.')
    return self._build_dataset(data_rng,
                               split,
                               data_dir,
                               global_batch_size,
                               cache,
                               repeat_final_dataset)

  @property
  def step_hint(self) -> int:
    """Max num steps the target setting algo was given to reach the target."""
    return 140_000
