"""LM workload parent class."""

import abc
import math
import os
from typing import Any, Dict, Iterator, Optional

import jax
import numpy as np
from absl import flags

from algoperf import spec

FLAGS = flags.FLAGS

USE_PYTORCH_DDP = 'LOCAL_RANK' in os.environ


class BaseLmWorkload(spec.Workload):
  """LM workload."""

  _vocab_size: int = 50257
  _seq_len: int = 1024
  _emb_dim: int = 1024
  _n_heads: int = 8
  _n_layers: int = 12
  _mlp_dim: int = 4096
  warmup_factor: float = 0.1

  # Model configuration
  _rmsnorm_epsilon: float = 1e-6
  _qknorm_epsilon: float = 1e-6
  _tie_embeddings: bool = True

  # Dtype configuration (as strings, to be converted by framework-specific subclasses)
  _compute_dtype_str: str = 'bfloat16'
  _param_dtype_str: str = 'float32'
  _output_dtype_str: str = 'bfloat16'  # Only used by JAX

  def __init__(self) -> None:
    super().__init__()
    self._param_shapes = None
    self._param_types = None

  @property
  def target_metric_name(self) -> str:
    """The name of the target metric (useful for scoring/processing code)."""
    return 'ppl'

  def has_reached_validation_target(self, eval_result: float) -> bool:
    return eval_result['validation/ppl'] <= self.validation_target_value

  @property
  def validation_target_value(self) -> float:
    return 22.432  # Target perplexity

  def has_reached_test_target(self, eval_result: Dict[str, float]) -> bool:
    return True  # No test targets

  @property
  def test_target_value(self) -> float:
    return None  # No test targets

  @property
  def loss_type(self) -> spec.LossType:
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self) -> int:
    return 8_749_870  # sequences of 1024 tokens each

  @property
  def num_eval_train_examples(self) -> int:
    return 10_000  # Subset for evaluation.

  @property
  def num_validation_examples(self) -> int:
    return 100_000  # sequences

  @property
  def num_test_examples(self) -> int:
    return 0

  @property
  def eval_batch_size(self) -> int:
    return 256

  @property
  def train_mean(self):
    raise NotImplementedError

  @property
  def train_stddev(self):
    raise NotImplementedError

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 31_967  # 8.9 hours

  @property
  def eval_period_time_sec(self) -> int:
    return 2_571  # approximately 25 evals

  @property
  def step_hint(self) -> int:
    """Approx. steps the baseline can do in the allowed runtime budget."""
    return 72_000

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
  def _build_input_queue(
    self,
    data_rng: jax.random.PRNGKey,
    split: str,
    data_dir: str,
    global_batch_size: int,
    cache: Optional[bool] = None,
    repeat_final_dataset: Optional[bool] = None,
    num_batches: Optional[int] = None,
  ) -> Iterator[Dict[str, Any]]:
    """Build an input queue for the given split."""

  @abc.abstractmethod
  def _eval_batch(
    self,
    params: spec.ParameterContainer,
    eval_batch: Dict[str, spec.Tensor],
    model_state: spec.ModelAuxiliaryState,
    rng: spec.RandomState,
  ) -> Dict[str, float]:
    """Evaluate the model on a single batch."""

  def _eval_model_on_split(
    self,
    split: str,
    num_examples: int,
    global_batch_size: int,
    params: spec.ParameterContainer,
    model_state: spec.ModelAuxiliaryState,
    rng: spec.RandomState,
    data_dir: str,
    global_step: int = 0,
  ) -> Dict[str, float]:
    """Run a full evaluation of the model."""
    num_batches = int(math.ceil(num_examples / global_batch_size))

    # Handle edge case where num_batches is 0 (e.g., test split with 0 examples)
    if num_batches == 0:
      return {'loss': 0.0, 'ppl': 1.0}

    if split not in self._eval_iters:
      # These iterators will repeat indefinitely.
      self._eval_iters[split] = self._build_input_queue(
        rng, split, data_dir, global_batch_size, num_batches=num_batches
      )

    eval_metrics = {}
    for _ in range(num_batches):
      eval_batch = next(self._eval_iters[split])
      metrics = self._eval_batch(params, eval_batch, model_state, rng)
      for metric_name, metric_value in metrics.items():
        eval_metrics.update(
          {metric_name: eval_metrics.get(metric_name, 0.0) + metric_value}
        )
      print(
        f"Completed eval batch {_ + 1}/{num_batches} for split '{split}' at global step {global_step}."
      )

    eval_results = self._normalize_eval_metrics(num_examples, eval_metrics)
    eval_results['ppl'] = np.exp(eval_results['loss']).item()
    return eval_results

  @abc.abstractmethod
  def _normalize_eval_metrics(
    self, num_examples: int, total_metrics: Dict[str, Any]
  ) -> Dict[str, float]:
    """Normalize eval metrics."""

  @abc.abstractmethod
  def loss_fn(
    self,
    label_batch: spec.Tensor,
    logits_batch: spec.Tensor,
    mask_batch: Optional[spec.Tensor] = None,
    label_smoothing: float = 0.0,
  ) -> Dict[str, spec.Tensor]:
    """Compute cross-entropy loss for language modeling."""

  def is_output_params(self, param_name: str) -> bool:
    """Return whether the given parameter is an output parameter."""
    return param_name.contains('output')
