"""FastMRI workload parent class."""

from algorithmic_efficiency import spec


class BaseFastMRIWorkload(spec.Workload):

  def __init__(self):
    self._param_shapes = None

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['validation/ssim'] > self.target_value

  @property
  def target_value(self):
    return 0.76

  @property
  def loss_type(self):
    return spec.LossType.MEAN_ABSOLUTE_ERROR

  @property
  def num_train_examples(self):
    return 34742

  @property
  def num_eval_train_examples(self):
    return 7135

  @property
  def num_validation_examples(self):
    return 7135

  @property
  def num_test_examples(self):
    return None

  @property
  def train_mean(self):
    return [0., 0., 0.]

  @property
  def train_stddev(self):
    return [1., 1., 1.]

  # data augmentation settings

  @property
  def center_fractions(self):
    return (0.08,)

  @property
  def aspect_ratio_range(self):
    return (0.75, 4.0 / 3.0)

  @property
  def accelerations(self):
    return (4,)

  @property
  def max_allowed_runtime_sec(self):
    return 111600  # 31 hours

  @property
  def eval_period_time_sec(self):
    return 6000  # 100 mins

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
