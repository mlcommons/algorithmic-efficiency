"""CIFAR workload parent class."""

import math
from typing import Dict, Tuple

from absl import flags
import jax
import torch
import torch.distributed as dist

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
import algorithmic_efficiency.random_utils as prng

FLAGS = flags.FLAGS
USE_PYTORCH_DDP, _, _, _ = pytorch_setup()


class BaseCifarWorkload(spec.Workload):

  def has_reached_validation_target(self, eval_result: Dict[str,
                                                            float]) -> bool:
    return eval_result['validation/accuracy'] > self.validation_target_value

  @property
  def validation_target_value(self) -> float:
    return 0.85

  def has_reached_test_target(self, eval_result: Dict[str, float]) -> bool:
    return eval_result['test/accuracy'] > self.validation_target_value

  @property
  def test_target_value(self) -> float:
    return 0.85

  @property
  def loss_type(self) -> spec.LossType:
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self) -> int:
    return 45000

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
    return 5000

  @property
  def num_test_examples(self) -> int:
    return 10000

  @property
  def eval_batch_size(self) -> int:
    return 1024

  @property
  def train_mean(self) -> Tuple[float, float, float]:
    return (0.49139968 * 255, 0.48215827 * 255, 0.44653124 * 255)

  @property
  def train_stddev(self) -> Tuple[float, float, float]:
    return (0.24703233 * 255, 0.24348505 * 255, 0.26158768 * 255)

  # Data augmentation settings.
  @property
  def scale_ratio_range(self) -> Tuple[float, float]:
    return (0.08, 1.0)

  @property
  def aspect_ratio_range(self) -> Tuple[float, float]:
    return (0.75, 4.0 / 3.0)

  @property
  def center_crop_size(self) -> int:
    return 32

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 3600  # 1 hour.

  @property
  def eval_period_time_sec(self) -> int:
    return 600  # 10 mins.

  @property
  def step_hint(self) -> int:
    # Note that the target setting algorithms were not actually run on this
    # workload, but for completeness we provide the number of steps for 100
    # epochs at batch size 1024.
    return 4883

  def _eval_model(
      self,
      params: spec.ParameterContainer,
      batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      rng: spec.RandomState) -> Dict[spec.Tensor, spec.ModelAuxiliaryState]:
    raise NotImplementedError

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
          data_rng=data_rng,
          split=split,
          data_dir=data_dir,
          global_batch_size=global_batch_size,
          cache=True,
          repeat_final_dataset=True)

    num_batches = int(math.ceil(num_examples / global_batch_size))
    num_devices = max(torch.cuda.device_count(), jax.local_device_count())
    eval_metrics = {}
    for _ in range(num_batches):
      batch = next(self._eval_iters[split])
      per_device_model_rngs = prng.split(model_rng, num_devices)
      # We already average these metrics across devices inside _compute_metrics.
      synced_metrics = self._eval_model(params,
                                        batch,
                                        model_state,
                                        per_device_model_rngs)
      for metric_name, metric_value in synced_metrics.items():
        if metric_name not in eval_metrics:
          eval_metrics[metric_name] = 0.0
        eval_metrics[metric_name] += metric_value

    if FLAGS.framework == 'jax':
      eval_metrics = jax.tree_map(lambda x: float(x[0] / num_examples),
                                  eval_metrics)
      return eval_metrics
    elif USE_PYTORCH_DDP:
      for metric in eval_metrics.values():
        dist.all_reduce(metric)
    return {k: float(v.item() / num_examples) for k, v in eval_metrics.items()}
