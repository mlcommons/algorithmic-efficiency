import itertools
import math
from typing import Dict, Optional

from absl import flags
import jax
import jax.numpy as jnp
import flax.linen as nn

from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech import input_pipeline
from algorithmic_efficiency.workloads.librispeech import metrics

FLAGS = flags.FLAGS


class BaseLibrispeechWorkload(spec.Workload):

  def __init__(self) -> None:
    self._eval_iters = {}
    self._param_shapes = None
    self._param_types = None
    self._num_outputs = 1024

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['validation/wer'] > self.target_value

  @property
  def target_value(self):
    return 0.10

  @property
  def loss_type(self):
    return spec.LossType.CTC_LOSS

  @property
  def num_train_examples(self):
    return 281241

  @property
  def num_eval_train_examples(self):
    return 2048

  @property
  def num_validation_examples(self):
    return 5567

  @property
  def num_test_examples(self):
    return 2620

  @property
  def train_mean(self):
    return 0.0

  @property
  def train_stddev(self):
    return 1.0

  @property
  def max_allowed_runtime_sec(self):
    return 36000  # 10h

  @property
  def eval_period_time_sec(self):
    return 500

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
      split = f'train_clean100[:{self.num_eval_train_examples}]'
    dataset_iter = input_pipeline.get_dataset_iter(split,
                                                   data_rng,
                                                   data_dir,
                                                   global_batch_size)
    return dataset_iter

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      logits: spec.Tensor,
      logit_paddings: spec.Tensor,
      targets: spec.Tensor,
      target_paddings: spec.Tensor) -> spec.Tensor:  # differentiable
    logprobs = nn.log_softmax(logits)
    per_seq_loss = self._ctc_loss(logprobs, logit_paddings, targets,
                                  target_paddings)
    normalizer = jnp.sum(1 - target_paddings)

    normalized_loss = jnp.sum(per_seq_loss) / jnp.maximum(normalizer, 1)
    return normalized_loss

  def _eval_metric(self, logits, logit_paddings, targets, target_paddings):
    normalized_loss = self.loss_fn(logits, logit_paddings, targets,
                                   target_paddings)
    return metrics.EvalMetrics.single_from_model_output(
        normalized_loss=normalized_loss)

  # Return whether or not a key in spec.ParameterContainer is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  # Keep this separate from the loss function in order to support optimizers
  # that use the logits.
  def output_activation_fn(self,
                           logits: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:
    pass

  def _eval_batch(self, params, batch, model_state, rng):
    (logits, logit_paddings), _ = self.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.EVAL,
        rng,
        update_batch_norm=False)
    return self._eval_metric(logits, logit_paddings, batch['targets'],
                             batch['target_paddings'])

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
      eval_iter = self.build_input_queue(
          data_rng, split, data_dir, global_batch_size=global_batch_size)
      # Note that this stores the entire val dataset in memory.
      self._eval_iters[split] = itertools.cycle(eval_iter)

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
