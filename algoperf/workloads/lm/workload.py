"""LM workload parent class."""

import abc
import math
import os
from typing import Dict, Optional

from absl import flags
import jax
import torch.distributed as dist

from algoperf import spec
from algoperf.workloads.lm import input_pipeline
from algoperf.workloads.lm.input_pipeline import get_hf_dataloader

FLAGS = flags.FLAGS

USE_PYTORCH_DDP = 'LOCAL_RANK' in os.environ


class BaseLmWorkload(spec.Workload):
  """LM workload."""

  _vocab_size: int = 50257
  _seq_len: int = 5
  warmup_factor: float = 0.1

  def __init__(self) -> None:
    super().__init__()
    self._param_shapes = None
    self._param_types = None

  @property
  def target_metric_name(self) -> str:
    """The name of the target metric (useful for scoring/processing code)."""
    return 'ppl'

  def has_reached_validation_target(self, eval_result: float) -> bool:
    return eval_result['validation/ppl'] > self.validation_target_value

  @property
  def validation_target_value(self) -> float:
    return 20.0  # Target perplexity

  def has_reached_test_target(self, eval_result: Dict[str, float]) -> bool:
    return eval_result['test/ppl'] <= self.test_target_value

  @property
  def test_target_value(self) -> float:
    return 20.0  # Target perplexity

  @property
  def loss_type(self) -> spec.LossType:
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self) -> int:
    return 1000000  # Example size

  @property
  def num_eval_train_examples(self) -> int:
    return 10000  # Subset for evaluation

  @property
  def num_validation_examples(self) -> int:
    return 50000 

  @property
  def num_test_examples(self) -> int:
    return 50000

  @property
  def eval_batch_size(self) -> int:
    return 8

  @property
  def train_mean(self):
    raise NotImplementedError

  @property
  def train_stddev(self):
    raise NotImplementedError

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 3600 * 4  # 4 hours

  @property
  def eval_period_time_sec(self) -> int:
    return 600  # 10 minutes

  @property
  def step_hint(self) -> int:
    """Approx. steps the baseline can do in the allowed runtime budget."""
    return 100000

  @property
  def pre_ln(self) -> bool:
    return True

  @property
  def attention_temp(self) -> float:
    return 1.0

  @property
  def activation(self) -> str:
    return 'silu'

  @property
  def glu(self) -> bool:
    return True

  @abc.abstractmethod
  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         num_batches: Optional[int] = None,
                         repeat_final_dataset: bool = False):
    """Build an input queue for the given split."""

  def _eval_batch(self,
                  params: spec.ParameterContainer,
                  batch: Dict[str, spec.Tensor],
                  model_state: spec.ModelAuxiliaryState,
                  rng: spec.RandomState) -> spec.Tensor:
    """Evaluate the model on a single batch."""
    logits, _ = self.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.EVAL,
        rng,
        update_batch_norm=False)
    
    loss_dict = self.loss_fn(batch['targets'], logits)
    return loss_dict['summed']

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
      loss += self._eval_batch(params, eval_batch, model_state, rng)
    if USE_PYTORCH_DDP:
      dist.all_reduce(loss)
    mean_loss = loss.item() / num_examples
    return {'loss': mean_loss}

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  @abc.abstractmethod
  def loss_fn(
      self,
      label_batch: spec.Tensor,
      logits_batch: spec.Tensor,
      mask_batch: Optional[spec.Tensor] = None,
      label_smoothing: float = 0.0) -> Dict[str, spec.Tensor]:
    """Compute cross-entropy loss for language modeling."""
