"""FastMRI workload parent class."""

import math
from typing import Optional

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.fastmri import input_pipeline


class BaseFastMRIWorkload(spec.Workload):

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['validation/ssim'] > self.target_value

  @property
  def target_value(self) -> float:
    return 0.7351

  @property
  def loss_type(self) -> spec.LossType:
    return spec.LossType.MEAN_ABSOLUTE_ERROR

  @property
  def num_train_examples(self) -> int:
    return 34742

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
    return 3554

  @property
  def num_test_examples(self) -> int:
    return 3581

  @property
  def eval_batch_size(self) -> int:
    return 256

  @property
  def train_mean(self):
    return [0., 0., 0.]

  @property
  def train_stddev(self):
    return [1., 1., 1.]

  @property
  def center_fractions(self):
    return (0.08,)

  @property
  def accelerations(self):
    return (4,)

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 10800  # 3 hours

  @property
  def eval_period_time_sec(self) -> int:
    return 80

  @property
  def step_hint(self) -> int:
    """Max num steps the target setting algo was given to reach the target."""
    return 27142

  def _build_input_queue(self,
                         data_rng: spec.RandomState,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         cache: Optional[bool] = None,
                         repeat_final_dataset: Optional[bool] = None,
                         num_batches: Optional[int] = None):
    del cache
    return input_pipeline.load_fastmri_split(global_batch_size,
                                             split,
                                             data_dir,
                                             data_rng,
                                             num_batches,
                                             repeat_final_dataset)
