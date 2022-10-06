import itertools
import math
from typing import Dict

from absl import flags
import jax

from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.ogbg import input_pipeline

FLAGS = flags.FLAGS


class BaseOgbgWorkload(spec.Workload):

  def __init__(self) -> None:
    self._eval_iters = {}
    self._param_shapes = None
    self._param_types = None
    self._num_outputs = 128

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['validation/mean_average_precision'] > self.target_value

  @property
  def target_value(self):
    # From Flax example
    # https://tensorboard.dev/experiment/AAJqfvgSRJaA1MBkc0jMWQ/#scalars.
    return 0.24

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self):
    return 350343

  @property
  def num_eval_train_examples(self):
    return 10000

  @property
  def num_validation_examples(self):
    return 43793

  @property
  def num_test_examples(self):
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
    return 120

  @property
  def param_shapes(self):
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    return self._param_shapes

  def build_input_queue(self,
                        data_rng: jax.random.PRNGKey,
                        split: str,
                        data_dir: str,
                        global_batch_size: int):
    if split == 'eval_train':
      split = f'train[:{self.num_eval_train_examples}]'
    dataset_iter = input_pipeline.get_dataset_iter(split,
                                                   data_rng,
                                                   data_dir,
                                                   global_batch_size)
    if split != 'train':
      # Note that this stores the entire val dataset in memory.
      dataset_iter = itertools.cycle(dataset_iter)
    return dataset_iter

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(self,
              label_batch: spec.Tensor,
              logits_batch: spec.Tensor,
              mask_batch: spec.Tensor,
              label_smoothing: float = 0.0) -> spec.Tensor:  # differentiable
    per_example_losses = self._binary_cross_entropy_with_mask(
        labels=label_batch,
        logits=logits_batch,
        mask=mask_batch,
        label_smoothing=label_smoothing)
    return per_example_losses

  # Return whether or not a key in spec.ParameterContainer is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  # Keep this separate from the loss function in order to support optimizers
  # that use the logits.
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
        dropout_rate=0.1,  # Unused for eval.
        aux_dropout_rate=None,
        update_batch_norm=False)
    return self._eval_metric(batch['targets'], logits, batch['weights'])

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str,
                           global_step: int = 0) -> Dict[str, float]:
    """Run a full evaluation of the model."""
    del global_step
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
