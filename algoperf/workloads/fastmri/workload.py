"""FastMRI workload parent class."""

import math
from typing import Optional

from algoperf import spec
from algoperf.workloads.fastmri import input_pipeline


class BaseFastMRIWorkload(spec.Workload):

  @property
  def target_metric_name(self) -> str:
    """The name of the target metric (useful for scoring/processing code)."""
    return 'ssim'

  @property
  def use_layer_norm(self) -> bool:
    """Whether or not to use LayerNorm in the model."""
    return False

  @property
  def use_tanh(self) -> bool:
    """Whether or not to use tanh activations in the model."""
    return False

  @property
  def num_pool_layers(self) -> bool:
    """Number of pooling layers."""
    return 4

  @property
  def num_channels(self) -> bool:
    """Number of channels."""
    return 32

  def has_reached_validation_target(self, eval_result: float) -> bool:
    return eval_result['validation/ssim'] > self.validation_target_value

  @property
  def validation_target_value(self) -> float:
    return 0.723653

  def has_reached_test_target(self, eval_result: float) -> bool:
    return eval_result['test/ssim'] > self.test_target_value

  @property
  def test_target_value(self) -> float:
    return 0.740633

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
    raise NotImplementedError

  @property
  def train_stddev(self):
    raise NotImplementedError

  @property
  def center_fractions(self):
    return (0.08,)

  @property
  def accelerations(self):
    return (4,)

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 4_430  # ~1.2 hours

  @property
  def eval_period_time_sec(self) -> int:
    return 80

  @property
  def step_hint(self) -> int:
    """Max num steps the baseline algo was given to reach the target."""
    return 36_189

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
