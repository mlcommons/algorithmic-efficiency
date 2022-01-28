from algorithmic_efficiency import spec


class ImagenetWorkload(spec.Workload):

  def __init__(self):
    self._eval_ds = None

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['accuracy'] > self.target_value

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
  def num_eval_examples(self):
    return 50000

  @property
  def train_mean(self):
    return [0.485 * 255, 0.456 * 255, 0.406 * 255]

  @property
  def train_stddev(self):
    return [0.229 * 255, 0.224 * 255, 0.225 * 255]

  """ data augmentation settings """

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

  def model_params_types(self):
    pass

  # Return whether or not a key in spec.ParameterTree is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    raise NotImplementedError

  def build_input_queue(self, data_rng: spec.RandomState, split: str,
                        data_dir: str, batch_size: int):
    return iter(self._build_dataset(data_rng, split, data_dir, batch_size))
