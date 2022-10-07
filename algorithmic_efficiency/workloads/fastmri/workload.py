"""FastMRI workload parent class."""

from typing import Optional

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.fastmri import input_pipeline


class BaseFastMRIWorkload(spec.Workload):

  def __init__(self):
    self._param_shapes = None
    self._param_types = None
    self._eval_iters = {}

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['validation/ssim'] > self.target_value

  @property
  def target_value(self):
    return 0.70

  @property
  def loss_type(self):
    return spec.LossType.MEAN_ABSOLUTE_ERROR

  @property
  def num_train_examples(self):
    return 34742

  @property
  def num_eval_train_examples(self):
    return 3474

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

  @property
  def center_fractions(self):
    return (0.08,)

  @property
  def accelerations(self):
    return (4,)

  @property
  def max_allowed_runtime_sec(self):
    return 10800  # 3 hours

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
