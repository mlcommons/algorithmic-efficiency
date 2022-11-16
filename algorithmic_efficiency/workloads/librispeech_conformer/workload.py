import math
from typing import Optional

from absl import flags
import jax
import torch 

from algorithmic_efficiency import data_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.libri_dataset import \
    LibriSpeechDataset

FLAGS = flags.FLAGS


class BaseLibrispeechWorkload(spec.Workload):

  _num_outputs: int = 1024

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['validation/wer'] < self.target_value

  @property
  def target_value(self) -> float:
    return 0.0842

  @property
  def loss_type(self) -> spec.LossType:
    return spec.LossType.CTC_LOSS

  @property
  def num_train_examples(self) -> int:
    return 263840

  @property
  def num_eval_train_examples(self) -> int:
    # Round up from num_validation_examples (which is the default for
    # num_eval_train_examples) to the next multiple of eval_batch_size, so that
    # we don't have to extract the correctly sized subset of the training data.
    rounded_up_multiple = math.ceil(self.num_validation_examples /
                                    self.eval_batch_size)
    return rounded_up_multiple * self.eval_batch_size

  @property
  def num_validation_examples(self) -> int:
    return 5348

  @property
  def num_test_examples(self) -> int:
    return 2472

  @property
  def eval_batch_size(self) -> int:
    return 256

  @property
  def train_mean(self) -> float:
    return 0.0

  @property
  def train_stddev(self) -> float:
    return 1.0

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 72000  # 20h

  @property
  def eval_period_time_sec(self) -> int:
    return 40 * 60  # 40m

  def _build_input_queue(self,
                         data_rng: spec.RandomState,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         cache: Optional[bool] = False,
                         repeat_final_dataset: Optional[bool] = False,
                         num_batches: Optional[int] = None):
    del cache
    del repeat_final_dataset
    return self._build_dataset(data_rng,
                               split,
                               data_dir,
                               global_batch_size,
                               num_batches)

  def _build_dataset(self,
                     data_rng: spec.RandomState,
                     split: str,
                     data_dir: str,
                     global_batch_size: int,
                     num_batches: Optional[int] = None):
    train = False
    if split == 'train':
      split = 'train-clean-100+train-clean-360+train-other-500'
      train = True
    elif split == 'eval_train':
      split = 'train-clean-100+train-clean-360+train-other-500'
    elif split == 'validation':
      split = 'dev-clean+dev-other'
    elif split == 'test':
      split = 'test-clean'

    ds = LibriSpeechDataset(split=split, data_dir=data_dir)
    ds_iter_batch_size = global_batch_size
    sampler = None

    dataloader = torch.utils.data.DataLoader(
      ds,
      batch_size=ds_iter_batch_size,
      shuffle=train,
      sampler=sampler,
      num_workers=4,
      prefetch_factor=10, 
      pin_memory=False,
      drop_last=train,)

    dataloader = data_utils.cycle(
      dataloader, custom_sampler=False, use_mixup=False)
    
    for batch in iter(dataloader):
      inputs, input_paddings = batch['inputs']
      targets, target_paddings = batch['targets']

      numpy_batch =  {
        'inputs': (inputs.numpy(), input_paddings.numpy()),
        'targets': (targets.numpy(), target_paddings.numpy()),
      }

      padded_batch = data_utils.shard_and_maybe_pad_np(numpy_batch, padding_value=1.0, global_batch_size=global_batch_size)
      yield padded_batch

  @property
  def step_hint(self) -> int:
    """Max num steps the target setting algo was given to reach the target."""
    return 100_000
