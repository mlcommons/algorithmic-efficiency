import math
from typing import Dict, Optional

import jax
import numpy as np
import torch

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.wmt import decode
from algorithmic_efficiency.workloads.wmt import input_pipeline

VOCAB_PATH = './wmt_256/sentencepiece_model'
WORKDIR = './wmt_256'


class BaseWmtWorkload(spec.Workload):
  """A WMT workload."""

  def __init__(self):
    self._eval_iters = {}
    self._param_shapes = None
    self._param_types = None
    self._tokenizer = None
    self._vocab_size = 32000

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
    return None

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
    for batch in iter(ds):
      batch = jax.tree_map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access
      yield batch

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
    eval_metrics_sums = {k: 0.0 for k in eval_metrics[0].keys()}
    for m in eval_metrics:
      for k, v in m.items():
        eval_metrics_sums[k] += v
    eval_denominator = eval_metrics_sums.pop("denominator")
    eval_results = jax.tree_map(
        lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
        eval_metrics_sums)

    bleu_score = self.translate_and_calculate_bleu(
        params=params,
        ds_iter=self._eval_iters[split],
        num_batches=num_batches,
        max_predict_length=256)

    eval_results['bleu'] = bleu_score
    return eval_results

  def compute_summed_metrics(self, logits, labels, weights):
    """Compute metrics summed across examples."""
    loss = self.compute_weighted_cross_entropy(logits, labels, weights, 0.0)
    acc_sum, weight_sum = self.compute_weighted_accuracy(
        logits, labels, weights)
    return {
        'loss': loss.sum(),
        'accuracy': acc_sum,
        'denominator': weight_sum,
    }

  def compute_weighted_accuracy(self, logits, targets, weights):
    """Compute weighted accuracy for log probs and targets.

    Args:
      logits: [batch, length, num_classes] float array.
      targets: categorical targets [batch, length] int array.
      weights: array of shape [batch, length]

    Returns:
      Tuple of scalar summed accuracy and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
      raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                       (str(logits.shape), str(targets.shape)))
    accuracy = (logits.argmax(-1) == targets) * weights
    normalizing_factor = weights.sum()
    return accuracy.sum(), normalizing_factor

  def _decode_tokens(self, toks):
    if isinstance(toks, torch.Tensor):
      toks = toks.cpu().numpy()
    valid_toks = toks[:np.argmax(toks == decode.EOS_ID) + 1].astype(np.int32)
    return self._tokenizer.detokenize(valid_toks).numpy().decode('utf-8')

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
      self._param_types = param_utils.jax_param_types(self._param_shapes)
    return self._param_types

  def output_activation_fn(self,
                           logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:
    """Return the final activations of the model."""
    pass

  def loss_fn(
      self,
      label_batch: spec.Tensor,  # Dense (not one-hot) labels.
      logits_batch: spec.Tensor,
      mask_batch: Optional[spec.Tensor] = None) -> spec.Tensor:
    del mask_batch
    return self.compute_weighted_cross_entropy(logits_batch, label_batch)
