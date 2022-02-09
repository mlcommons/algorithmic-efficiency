from algorithmic_efficiency import spec


class OGB(spec.Workload):

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['mean_average_precision'] > self.target_value

  @property
  def target_value(self):
    return 0.25

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
