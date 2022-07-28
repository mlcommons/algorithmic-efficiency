"""CIFAR workload parent class."""

from algorithmic_efficiency import spec


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
    return 3600  # 1 hours

  @property
  def eval_period_time_sec(self):
    return 600  # 10 mins

  @property
  def param_shapes(self):
    """The shapes of the parameters in the workload model."""
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    return self._param_shapes

  @property
  def model_params_types(self):
    """
    TODO: return shape tuples from model as a tree
    """
    raise NotImplementedError

  # Return whether or not a key in spec.ParameterTree is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    raise NotImplementedError
