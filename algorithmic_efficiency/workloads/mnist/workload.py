"""MNIST workload parent class."""
from typing import Dict, Tuple

import itertools
from algorithmic_efficiency import spec
import algorithmic_efficiency.random_utils as prng


class BaseMnistWorkload(spec.Workload):

  def __init__(self):
    self._eval_iters = {}

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['accuracy'] > self.target_value

  @property
  def target_value(self):
    return 0.9

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self):
    return 60000

  @property
  def num_eval_train_examples(self):
    return 10000

  @property
  def num_validation_examples(self):
    return 10000

  def num_test_examples(self):
    return 10000

  @property
  def train_mean(self):
    return 0.1307

  @property
  def train_stddev(self):
    return 0.3081

  @property
  def max_allowed_runtime_sec(self):
    return 60

  @property
  def eval_period_time_sec(self):
    return 10

  def _eval_model(
      self,
      params: spec.ParameterContainer,
      images: spec.Tensor,
      labels: spec.Tensor,
      model_state: spec.ModelAuxiliaryState,
      rng: spec.RandomState) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
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
    # DO NOT SUBMIT use num_examples
    data_rng, model_rng = prng.split(rng, 2)
    if split not in self._eval_iters:
      eval_iter = self.build_input_queue(
          data_rng, split, data_dir, global_batch_size=global_batch_size)
      # Note that this stores the entire val dataset in memory.
      self._eval_iters[split] = itertools.cycle(eval_iter)

    total_metrics = {
        'accuracy': 0.,
        'loss': 0.,
    }
    n_data = 0
    for (images, labels, _) in self._eval_iters[split]:
      batch_metrics = self._eval_model(
          params,
          images,
          labels,
          model_state,
          model_rng)
      total_metrics = {
          k: v + batch_metrics[k] for k, v in total_metrics.items()
      }
      n_data += batch_metrics['n_data']
    return {k: float(v / n_data) for k, v in total_metrics.items()}
