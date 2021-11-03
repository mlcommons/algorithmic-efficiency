import random_utils as prng
import spec


class OGB(spec.Workload):

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['average_precision'] > self.target_value

  @property
  def target_value(self):
    return 0.255

  @property
  def loss_type(self):
    return spec.LossType.SIGMOID_CROSS_ENTROPY

  @property
  def num_train_examples(self):
    return 350343

  @property
  def num_eval_examples(self):
    return 43793

  @property
  def train_mean(self):
    raise NotImplementedError

  @property
  def train_stddev(self):
    raise NotImplementedError

  @property
  def max_allowed_runtime_sec(self):
    return 12000  # 3h20m

  @property
  def eval_period_time_sec(self):
    return 360  # 60 minutes (too long) 

  def eval_model(
      self,
      params: spec.ParameterContainer,
      model_state: spec.ModelAuxiliaryState,
      rng: spec.RandomState,
      data_dir: str):
    """Run a full evaluation of the model."""
    data_rng, model_rng = prng.split(rng, 2)
    eval_batch_size = 256
    num_batches = self.num_eval_examples // eval_batch_size
    if self._eval_ds is None:
      self._eval_ds = self.build_input_queue(
          data_rng, 'test', data_dir, batch_size=eval_batch_size)

    self._model.deterministic = True

    total_metrics = {
        'accuracy': 0.,
        'average_precision': 0.,
        'loss': 0.,
    }
    # Loop over graphs.
    for graphs in self._eval_ds:
      logits, _ = self.model_fn(
          params,
          graphs,
          model_state,
          spec.ForwardPassMode.EVAL,
          model_rng,
          update_batch_norm=False)
      labels = graphs.globals
      batch_metrics = self._eval_metric(labels, logits)
      total_metrics = {
          k: v + batch_metrics[k] for k, v in total_metrics.items()
      }
    return {k: float(v / num_batches) for k, v in total_metrics.items()}
