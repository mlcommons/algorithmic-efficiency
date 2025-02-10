"""Criteo1TB DLRM workload base class."""

import math
import os
from typing import Dict, Iterator, Optional, Tuple

from absl import flags
import torch.distributed as dist

from algoperf import spec
from algoperf.workloads.criteo1tb import input_pipeline

FLAGS = flags.FLAGS

USE_PYTORCH_DDP = 'LOCAL_RANK' in os.environ


class BaseCriteo1TbDlrmSmallWorkload(spec.Workload):
  """Criteo1tb workload."""

  vocab_size: int = 32 * 128 * 1024  # 4_194_304.
  num_dense_features: int = 13
  mlp_bottom_dims: Tuple[int, int] = (512, 256, 128)
  mlp_top_dims: Tuple[int, int, int] = (1024, 1024, 512, 256, 1)
  embed_dim: int = 128

  @property
  def target_metric_name(self) -> str:
    """The name of the target metric (useful for scoring/processing code)."""
    return 'loss'

  def has_reached_validation_target(self, eval_result: Dict[str,
                                                            float]) -> bool:
    return eval_result['validation/loss'] < self.validation_target_value

  @property
  def use_layer_norm(self) -> bool:
    """Whether or not to use LayerNorm in the model."""
    return False

  @property
  def use_resnet(self) -> bool:
    """Whether or not to use residual connections in the model."""
    return False

  @property
  def embedding_init_multiplier(self) -> float:
    return None

  @property
  def validation_target_value(self) -> float:
    return 0.123735

  def has_reached_test_target(self, eval_result: Dict[str, float]) -> bool:
    return eval_result['test/loss'] < self.test_target_value

  @property
  def test_target_value(self) -> float:
    return 0.126041

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
  def num_validation_examples(self) -> int:
    return 83_274_637

  @property
  def num_test_examples(self) -> int:
    return 95_000_000

  @property
  def train_mean(self):
    raise NotImplementedError

  @property
  def train_stddev(self):
    raise NotImplementedError

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 7703  # ~2 hours.

  @property
  def eval_period_time_sec(self) -> int:
    return 2 * 60  # 2 mins.

  def _build_input_queue(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      global_batch_size: int,
      cache: Optional[bool] = None,
      repeat_final_dataset: Optional[bool] = None,
      num_batches: Optional[int] = None) -> Iterator[Dict[str, spec.Tensor]]:
    del cache
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
    """Max num steps the baseline algo was given to reach the target."""
    return 10_666

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
          data_rng=rng,
          split=split,
          data_dir=data_dir,
          global_batch_size=global_batch_size,
          num_batches=num_batches,
          repeat_final_dataset=True)
    loss = 0.0
    for _ in range(num_batches):
      eval_batch = next(self._eval_iters[split])
      loss += self._eval_batch(params, eval_batch)
    if USE_PYTORCH_DDP:
      dist.all_reduce(loss)
    mean_loss = loss.item() / num_examples
    return {'loss': mean_loss}
