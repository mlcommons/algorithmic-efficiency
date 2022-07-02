"""Cifar workload parent class."""

import itertools
from typing import Dict, Tuple

import jax

from algorithmic_efficiency import spec
import algorithmic_efficiency.random_utils as prng


class BaseCifarWorkload(spec.Workload):

  def __init__(self):
    self._eval_iters = {}
    self._param_shapes = None
    self._param_types = None

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['validation/accuracy'] > self.target_value

  @property
  def target_value(self):
    return 0.85

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self):
    return 45000

  @property
  def num_eval_train_examples(self):
    return 10000

  @property
  def num_validation_examples(self):
    return 5000

  @property
  def num_test_examples(self):
    return 10000

  @property
  def train_mean(self):
    return [0.49139968 * 255, 0.48215827 * 255, 0.44653124 * 255]

  @property
  def train_stddev(self):
    return [0.24703233 * 255, 0.24348505 * 255, 0.26158768 * 255]

  # data augmentation settings
  @property
  def scale_ratio_range(self):
    return (0.08, 1.0)

  @property
  def aspect_ratio_range(self):
    return (0.75, 4.0 / 3.0)

  @property
  def center_crop_size(self):
    return 32

  @property
  def max_allowed_runtime_sec(self):
    return 3600

  @property
  def eval_period_time_sec(self):
    return 600

  def _eval_model(
      self,
      params: spec.ParameterContainer,
      images: spec.Tensor,
      labels: spec.Tensor,
      model_state: spec.ModelAuxiliaryState,
      rng: spec.RandomState) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    """Return the summed metrics for a given batch."""
    raise NotImplementedError

  def _eval_metric(self, logits, labels):
    """Return the mean accuracy and loss as a dict."""
    raise NotImplementedError

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str) -> Dict[str, float]:
    """Run a full evaluation of the model."""
    data_rng, model_rng = prng.split(rng, 2)
    if split not in self._eval_iters:
      eval_iter = self.build_input_queue(
          data_rng, split, data_dir, global_batch_size=global_batch_size)
      # Note that this stores the entire eval dataset in memory.
      self._eval_iters[split] = itertools.cycle(eval_iter)

    total_metrics = {'accuracy': 0., 'loss': 0.}
    num_data = 0
    num_batches = num_examples // global_batch_size
    for bi, batch in enumerate(self._eval_iters[split]):
      if bi > num_batches:
        break
      per_device_model_rngs = prng.split(model_rng, jax.local_device_count())
      batch_metrics = self._eval_model(params,
                                       batch,
                                       model_state,
                                       per_device_model_rngs)
      total_metrics = {
          k: v + batch_metrics[k] for k, v in total_metrics.items()
      }
      num_data += batch_metrics['num_data']
    return {k: float(v / num_data) for k, v in total_metrics.items()}

  def build_input_queue(self,
                        data_rng: spec.RandomState,
                        split: str,
                        data_dir: str,
                        global_batch_size: int):
    """Build an input queue for the given split."""
    ds = self._build_dataset(data_rng, split, data_dir, global_batch_size)
    for batch in iter(ds):
      batch = jax.tree_map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access
      yield batch
