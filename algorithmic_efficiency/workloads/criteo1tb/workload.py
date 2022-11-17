import math
import os
from typing import Dict, Optional, Tuple

import jax
import torch.distributed as dist

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.criteo1tb import input_pipeline

USE_PYTORCH_DDP = 'LOCAL_RANK' in os.environ


class BaseCriteo1TbDlrmSmallWorkload(spec.Workload):
  """Criteo1tb workload."""

  vocab_size: int = 32 * 128 * 1024  # 4_194_304
  num_dense_features: int = 13
  mlp_bottom_dims: Tuple[int, int] = (512, 256, 128)
  mlp_top_dims: Tuple[int, int, int] = (1024, 1024, 512, 256, 1)
  embed_dim: int = 128

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['validation/loss'] < self.target_value

  @property
  def target_value(self):
    return 0.124225

  @property
  def loss_type(self) -> spec.LossType:
    return spec.LossType.SIGMOID_CROSS_ENTROPY

  @property
  def num_train_examples(self) -> int:
    return 4_195_197_692

  @property
  def num_eval_train_examples(self) -> int:
    # Round up from num_validation_examples (which is the default for
    # num_eval_train_examples) to the next multiple of eval_batch_size, so that
    # we don't have to extract the correctly sized subset of the training data.
    rounded_up_multiple = math.ceil(self.num_validation_examples /
                                    self.eval_batch_size)
    return rounded_up_multiple * self.eval_batch_size

  @property
  def num_validation_examples(self):
    return 89_000_000

  @property
  def num_test_examples(self):
    return 89_274_637

  @property
  def eval_batch_size(self) -> int:
    return 524_288

  @property
  def train_mean(self) -> float:
    return 0.0

  @property
  def train_stddev(self) -> float:
    return 1.0

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 6 * 60 * 60

  @property
  def eval_period_time_sec(self):
    return 9 * 60

  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         num_batches: Optional[int] = None,
                         repeat_final_dataset: bool = False):
    ds = input_pipeline.get_criteo1tb_dataset(
        split=split,
        shuffle_rng=data_rng,
        data_dir=data_dir,
        num_dense_features=self.num_dense_features,
        global_batch_size=global_batch_size,
        num_batches=num_batches,
        repeat_final_dataset=repeat_final_dataset)

    for batch in iter(ds):
      yield batch

  @property
  def step_hint(self) -> int:
    """Max num steps the target setting algo was given to reach the target."""
    return 8000

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
    del model_state
    del global_step
    num_batches = int(math.ceil(num_examples / global_batch_size))
    if split not in self._eval_iters:
      # These iterators will repeat indefinitely.
      self._eval_iters[split] = self._build_input_queue(
          rng,
          split,
          data_dir,
          global_batch_size,
          num_batches,
          repeat_final_dataset=True)
    loss = 0.0
    for _ in range(num_batches):
      eval_batch = next(self._eval_iters[split])
      loss += self._eval_batch(params, eval_batch)
    if USE_PYTORCH_DDP:
      dist.all_reduce(loss)
    mean_loss = loss.item() / num_examples
    return {'loss': mean_loss}
