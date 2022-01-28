from absl import logging
import random_utils as prng

from algorithmic_efficiency import spec


class Mnist(spec.Workload):

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
  def num_eval_examples(self):
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

  def _eval_metric(self, logits, labels):
    """Return the mean accuracy and loss as a dict."""
    raise NotImplementedError

  def eval_model(self, params: spec.ParameterContainer,
                 model_state: spec.ModelAuxiliaryState, rng: spec.RandomState,
                 data_dir: str):
    """Run a full evaluation of the model."""
    data_rng, model_rng = prng.split(rng, 2)
    eval_batch_size = 2000
    num_batches = self.num_eval_examples // eval_batch_size
    self._eval_ds = self.build_input_queue(
        data_rng, 'test', data_dir, batch_size=eval_batch_size)

    total_metrics = {
        'accuracy': 0.,
        'loss': 0.,
    }
    for (images, labels) in self._eval_ds:
      images, labels = self.preprocess_for_eval(images, labels, None, None)
      logits, _ = self.model_fn(
          params,
          images,
          model_state,
          spec.ForwardPassMode.EVAL,
          model_rng,
          update_batch_norm=False)
      # TODO(znado): add additional eval metrics?
      batch_metrics = self._eval_metric(logits, labels)
      total_metrics = {
          k: v + batch_metrics[k] for k, v in total_metrics.items()
      }
    return {k: float(v / num_batches) for k, v in total_metrics.items()}
