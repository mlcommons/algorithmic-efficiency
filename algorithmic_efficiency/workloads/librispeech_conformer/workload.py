from typing import Optional

from absl import flags
from absl import logging
import jax
import numpy as np
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech_conformer import \
    input_pipeline

FLAGS = flags.FLAGS


class BaseLibrispeechWorkload(spec.Workload):

  _num_outputs: int = 1024

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['validation/wer'] < self.target_value

  @property
  def target_value(self):
    return 0.0842

  @property
  def loss_type(self):
    return spec.LossType.CTC_LOSS

  @property
  def num_train_examples(self):
    return 263840

  @property
  def num_eval_train_examples(self):
    return 256

  @property
  def num_validation_examples(self):
    return 5348

  @property
  def num_test_examples(self):
    return 2472

  @property
  def eval_batch_size(self):
    return 256

  @property
  def train_mean(self):
    return 0.0

  @property
  def train_stddev(self):
    return 1.0

  @property
  def max_allowed_runtime_sec(self):
    return 108000  # 30h

  @property
  def eval_period_time_sec(self):
    return 40 * 60  # 40m

  def _build_input_queue(self,
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
                               num_batches)

  def shard(self, batch, n_devices=None):
    if n_devices is None:
      n_devices = max(torch.cuda.device_count(), jax.local_device_count())

    # Otherwise, the entries are arrays, so just reshape them.
    def _shard_array(array):
      return array.reshape((n_devices, -1) + array.shape[1:])

    return jax.tree_map(_shard_array, batch)

  def maybe_pad_batch(self, batch, desired_batch_size, padding_value=0.0):
    """Zero pad the batch on the right to desired_batch_size.

    All keys in the batch dictionary will have their corresponding arrays
    padded. Will return a dictionary with the same keys.

    Args:
      batch: A dictionary mapping keys to arrays. We assume that inputs is
      one of the keys.
      desired_batch_size: All arrays in the dict will be padded to have
      first dimension equal to desired_batch_size.
      padding_value: value to be used as padding.

    Returns:
      A dictionary mapping the same keys to the padded batches. Additionally
      we add a key representing weights, to indicate how the batch was padded.
    """
    batch_axis = 0
    inputs, input_paddings = batch['inputs']
    targets, target_paddings = batch['targets']

    batch_size = inputs.shape[batch_axis]
    batch_pad = desired_batch_size - batch_size

    # Most batches will not need padding so we quickly return to avoid slowdown.
    if batch_pad == 0:
      new_batch = jax.tree_map(lambda x: x, batch)
      return new_batch

    def zero_pad(ar, pad_axis):
      pw = [(0, 0)] * ar.ndim
      pw[pad_axis] = (0, batch_pad)
      return np.pad(ar, pw, mode='constant', constant_values=padding_value)

    padded_batch = {
        'inputs': (zero_pad(inputs, batch_axis),
                   zero_pad(input_paddings, batch_axis)),
        'targets': (zero_pad(targets, batch_axis),
                    zero_pad(target_paddings, batch_axis))
    }
    return padded_batch

  def _build_dataset(self,
                     data_rng: spec.RandomState,
                     split: str,
                     data_dir: str,
                     batch_size: int,
                     num_batches: Optional[int] = None):
    train = False
    if split == 'train':
      split = 'train-clean-100+train-clean-360+train-other-500'
      train = True
    elif split == 'eval_train':
      split = 'train-clean-100'
    elif split == 'validation':
      split = 'dev-clean+dev-other'
    elif split == 'test':
      split = 'test-clean'

    ds = input_pipeline.get_librispeech_dataset(split,
                                                data_dir,
                                                data_rng,
                                                train,
                                                batch_size,
                                                num_batches)

    logging.info('done loading split = %s', split)

    for batch in iter(ds):
      batch = jax.tree_map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access
      batch = self.maybe_pad_batch(batch, batch_size, padding_value=1.0)
      batch = self.shard(batch)

      yield batch

  @property
  def step_hint(self) -> int:
    """Max num steps the target setting algo was given to reach the target."""
    return 100_000
