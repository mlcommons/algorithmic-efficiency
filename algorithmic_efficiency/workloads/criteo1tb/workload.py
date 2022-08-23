import math
from typing import Dict, Optional

from absl import flags
import jax

from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
from algorithmic_efficiency.workloads.criteo1tb import input_pipeline

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()

FLAGS = flags.FLAGS


class BaseCriteo1TbDlrmSmallWorkload(spec.Workload):
  """Criteo1tb workload."""

  def __init__(self):
    self._eval_iters = {}
    self._param_shapes = None
    self._param_types = None

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['validation/loss'] < self.target_value

  @property
  def target_value(self):
    return 0.13

  @property
  def loss_type(self):
    return spec.LossType.SIGMOID_CROSS_ENTROPY

  @property
  def num_train_examples(self):
    return 4_195_197_692

  @property
  def num_eval_train_examples(self):
    return 4_195_197_692

  @property
  def num_validation_examples(self):
    return 89_137_318

  @property
  def num_test_examples(self):
    return None

  @property
  def train_mean(self):
    return 0.0

  @property
  def train_stddev(self):
    return 1.0

  @property
  def max_allowed_runtime_sec(self):
    return 6 * 60 * 60

  @property
  def eval_period_time_sec(self):
    return 20 * 60

  @property
  def num_dense_features(self):
    return 13

  @property
  def vocab_sizes(self):
    return tuple([1024] * 26)

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

  def build_input_queue(self,
                        data_rng: jax.random.PRNGKey,
                        split: str,
                        data_dir: str,
                        global_batch_size: int,
                        num_batches: Optional[int] = None,
                        repeat_final_dataset: bool = False):
    is_training = split == 'train'
    ds = input_pipeline.get_criteo1tb_dataset(
        split=split,
        data_dir=data_dir,
        is_training=is_training,
        global_batch_size=global_batch_size,
        num_dense_features=self.num_dense_features,
        vocab_sizes=self.vocab_sizes,
        num_batches=num_batches,
        repeat_final_dataset=repeat_final_dataset)
    for batch in iter(ds):
      batch = jax.tree_map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access
      yield batch

  # Return whether or not a key in spec.ParameterContainer is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def output_activation_fn(self,
                           logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:
    pass

  def _eval_batch(self, params, batch, model_state, rng):
    logits, _ = self.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.EVAL,
        rng,
        update_batch_norm=False)
    return self._eval_metric(batch['targets'], logits)

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
      self._eval_iters[split] = self.build_input_queue(
          data_rng, split, data_dir, global_batch_size=global_batch_size)

    total_metrics = None
    num_eval_steps = int(math.ceil(float(num_examples) / global_batch_size))
    # Loop over graph batches in eval dataset.
    for _ in range(num_eval_steps):
      batch = next(self._eval_iters[split])
      batch_metrics = self._eval_batch(params, batch, model_state, model_rng)
      total_metrics = (
          batch_metrics
          if total_metrics is None else total_metrics.merge(batch_metrics))
    if total_metrics is None:
      return {}
    if FLAGS.framework == 'jax':
      total_metrics = total_metrics.reduce()
    return {k: float(v) for k, v in total_metrics.compute().items()}
