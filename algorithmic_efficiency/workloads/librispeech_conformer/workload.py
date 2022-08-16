import itertools
import math
from typing import Dict, Optional

from absl import flags
import jax
import jax.numpy as jnp
import flax.linen as nn

from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech_conformer import input_pipeline
from algorithmic_efficiency.workloads.librispeech_conformer import metrics
import numpy as np 

import tensorflow_datasets as tfds
FLAGS = flags.FLAGS


class BaseLibrispeechWorkload(spec.Workload):

  def __init__(self) -> None:
    self._eval_iters = {}
    self._param_shapes = None
    self._param_types = None
    self._num_outputs = 1024

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['validation/ctc_loss'] < 0.5

  @property
  def target_value(self):
    return 0.10

  @property
  def loss_type(self):
    return spec.LossType.CTC_LOSS

  @property
  def num_train_examples(self):
    return 28539

  @property
  def num_eval_train_examples(self):
    return 64

  @property
  def num_validation_examples(self):
    return 2703

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
    return 200

  @property
  def param_shapes(self):
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    return self._param_shapes

  def build_input_queue(self,
                        data_rng: spec.RandomState,
                        split: str,
                        data_dir: str,
                        global_batch_size: int,
                        cache: Optional[bool] = False,
                        repeat_final_dataset: Optional[bool] = False,
                        num_batches: Optional[int] = None):
    return self._build_dataset(data_rng,
                               split,
                               data_dir,
                               global_batch_size,
                               cache,
                               repeat_final_dataset,
                               num_batches)

  def shard(self, batch, n_devices=None):    
    if n_devices is None:
      n_devices = jax.local_device_count()

    # Otherwise, the entries are arrays, so just reshape them.
    def _shard_array(array):
      return array.reshape((n_devices, -1) + array.shape[1:])

    return jax.tree_map(_shard_array, batch)

  def _build_dataset(self,
                     data_rng: spec.RandomState,
                     split: str,
                     data_dir: str,
                     batch_size: int,
                     cache: Optional[bool] = False,
                     repeat_final_dataset: Optional[bool] = False,
                     num_batches: Optional[int] = None):
    if batch_size % jax.local_device_count() > 0:
      raise ValueError('Batch size must be divisible by the number of devices')
    
    train = False

    if split == 'train':
      split = 'train-clean-100'
      train = True
    elif split == 'eval_train':
      split = 'train-clean-100'
    elif split=='validation':
      split = 'dev-clean'
    elif split == 'test':
      split = 'test-clean'
    
    if split is None:
      return None

    ds = input_pipeline.get_librispeech_dataset(
        split,
        data_dir,
        train,
        batch_size,
        num_batches=num_batches,
        repeat_final_dataset=repeat_final_dataset)
    for batch in iter(ds):
      batch = jax.tree_map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access
      batch = self.shard(batch)

      yield batch

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      logits: spec.Tensor,
      logit_paddings: spec.Tensor,
      targets: spec.Tensor,
      target_paddings: spec.Tensor) -> spec.Tensor:  # differentiable
    logprobs = nn.log_softmax(logits)
    per_seq_loss = self.ctc_loss(logprobs, logit_paddings, targets,
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
