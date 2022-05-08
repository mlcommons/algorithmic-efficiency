import math
from typing import Dict, Optional

from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency import param_utils
from algorithmic_efficiency.workloads.wmt import decode
from algorithmic_efficiency.workloads.wmt import input_pipeline

VOCAB_PATH = './wmt_256/sentencepiece_model'
WORKDIR = './wmt_256'


class BaseWmtWorkload(spec.Workload):
  """A WMT workload."""

  def __init__(self):
    self._eval_ds = None
    self._train_ds = None
    self._predict_ds = None
    self._encoder = None
    self._vocab_size = 32000
    self._global_batch_size = None
    self._param_shapes = None

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['validation/bleu'] > self.target_value

  @property
  def target_value(self):
    return 25

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self):
    return 5906184

  @property
  def num_eval_train_examples(self):
    return 3004

  @property
  def num_validation_examples(self):
    return 3004

  @property
  def num_test_examples(self):
    pass

  @property
  def train_mean(self):
    return 0.0

  @property
  def train_stddev(self):
    return 1.0

  @property
  def max_allowed_runtime_sec(self):
    return 80000

  @property
  def eval_period_time_sec(self):
    return 800

  def build_input_queue(self,
                        data_rng: jax.random.PRNGKey,
                        split: str,
                        data_dir: str,
                        global_batch_size: int,
                        num_batches: Optional[int] = None,
                        repeat_final_dataset: bool = False):
    is_training = split == 'train'
    if split == 'eval_train':
      split = 'train'
    ds, self._tokenizer = input_pipeline.get_wmt_dataset(
        data_rng,
        split,
        data_dir,
        is_training=is_training,
        vocab_size=self._vocab_size,
        global_batch_size=global_batch_size,
        num_batches=num_batches,
        reverse_translation=True,
        repeat_final_dataset=repeat_final_dataset)
    for eval_batch in iter(ds):
      eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
      yield eval_batch

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str) -> Dict[str, float]:
    """Run a full evaluation of the model."""
    num_batches = int(math.ceil(num_examples / global_batch_size))
    if split not in self._eval_iters:
      # These iterators will repeat indefinitely.
      self._eval_iters[split] = self.build_input_queue(
          rng,
          split,
          data_dir,
          global_batch_size,
          num_batches,
          repeat_final_dataset=True)
    eval_metrics = []
    for _ in range(num_batches):
      eval_batch = next(self._eval_iters[split])
      metrics = self.eval_step(params, eval_batch)
      eval_metrics.append(metrics)
    eval_metrics = common_utils.get_metrics(eval_metrics)
    eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
    eval_denominator = eval_metrics_sums.pop("denominator")
    eval_results = jax.tree_map(
        lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
        eval_metrics_sums)

    bleu_score = self.translate_and_calculate_bleu(
        target=params,
        ds_iter=self._eval_iters[split],
        num_batches=num_batches,
        decode_tokens=self._decode_tokens,
        max_predict_length=256)

    eval_results['bleu'] = bleu_score
    return eval_results

  def compute_metrics(self, logits, labels, weights):
    """Compute summary metrics."""
    loss = self.compute_weighted_cross_entropy(logits, labels, weights, 0.0)
    acc, weight_sum = self.compute_weighted_accuracy(logits, labels, weights)
    metrics = {
        'loss': loss.sum(),
        'accuracy': acc,
        'denominator': weight_sum,
    }
    return metrics

  def compute_weighted_accuracy(self, logits, targets, weights):
    """Compute weighted accuracy for log probs and targets.

    Args:
      logits: [batch, length, num_classes] float array.
      targets: categorical targets [batch, length] int array.
      weights: array of shape [batch, length]

    Returns:
      Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
      raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                       (str(logits.shape), str(targets.shape)))
    loss = (logits.argmax(-1) == targets) * weights
    normalizing_factor = weights.sum()
    return loss.sum(), normalizing_factor

  def _decode_tokens(self, toks):
    if isinstance(toks, torch.Tensor):
      toks = toks.cpu().numpy()
    valid_toks = toks[:np.argmax(toks == decode.EOS_ID) + 1].astype(np.int32)
    return self._encoder.detokenize(valid_toks).numpy().decode('utf-8')

  # Return whether or not a key in spec.ParameterContainer is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

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
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    if self._param_types is None:
      self._param_types = param_utils.jax_param_types(
          self._param_shapes.unfreeze())
    return self._param_types

  def output_activation_fn(self,
                           logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:
    """Return the final activations of the model."""
    pass
