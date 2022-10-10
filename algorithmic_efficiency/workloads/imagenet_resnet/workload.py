"""ImageNet workload parent class."""
from typing import Optional

import jax

from algorithmic_efficiency import spec


class BaseImagenetResNetWorkload(spec.Workload):

  def __init__(self):
    self._num_classes = 1000

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['validation/accuracy'] > self.target_value

  @property
  def target_value(self):
    return 0.76

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self):
    return 1281167

  @property
  def num_eval_train_examples(self):
    return 50000

  @property
  def num_validation_examples(self):
    return 50000

  @property
  def num_test_examples(self):
    return 10000  # ImageNet-v2

  @property
  def train_mean(self):
    return [0.485 * 255, 0.456 * 255, 0.406 * 255]

  @property
  def train_stddev(self):
    return [0.229 * 255, 0.224 * 255, 0.225 * 255]

  # Data augmentation settings.

  @property
  def scale_ratio_range(self):
    return (0.08, 1.0)

  @property
  def aspect_ratio_range(self):
    return (0.75, 4.0 / 3.0)

  @property
  def center_crop_size(self):
    return 224

  @property
  def resize_size(self):
    return 256

  @property
  def max_allowed_runtime_sec(self):
    return 111600  # 31 hours

  @property
  def eval_period_time_sec(self):
    return 6000  # 100 mins

  # Return whether or not a key in spec.ParameterTree is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    raise NotImplementedError

  def _build_input_queue(self,
                         data_rng: spec.RandomState,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         cache: Optional[bool] = None,
                         repeat_final_dataset: Optional[bool] = None,
                         num_batches: Optional[int] = None):
    del num_batches
    if global_batch_size % jax.local_device_count() != 0:
      raise ValueError('Batch size must be divisible by the number of devices')
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
