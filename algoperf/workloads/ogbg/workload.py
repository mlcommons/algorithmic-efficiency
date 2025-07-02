"""OGBG workload parent class."""

import abc
import itertools
import math
from typing import Any, Dict, Optional, Tuple

import jax

from algoperf import random_utils as prng
from algoperf import spec
from algoperf.workloads.ogbg import input_pipeline
from algoperf.workloads.ogbg import metrics

from operator import attrgetter


class BaseOgbgWorkload(spec.Workload):

  _num_outputs: int = 128

  @property
  def target_metric_name(self) -> str:
    """The name of the target metric (useful for scoring/processing code)."""
    return 'mean_average_precision'

  @property
  def activation_fn_name(self) -> str:
    """Name of the activation function to use. One of 'relu', 'gelu', 'silu'."""
    return 'relu'

  @property
  def hidden_dims(self) -> Tuple[int]:
    return (256,)

  @property
  def latent_dim(self) -> int:
    return 256

  @property
  def num_message_passing_steps(self) -> int:
    return 5

  def has_reached_validation_target(self, eval_result: float) -> bool:
    return False
    # return eval_result[
    #     'validation/mean_average_precision'] > self.validation_target_value

  @property
  def validation_target_value(self) -> float:
    return 0.28098

  def has_reached_test_target(self, eval_result: float) -> bool:
    return False
    # return eval_result['test/mean_average_precision'] > self.test_target_value

  @property
  def test_target_value(self) -> float:
    return 0.268729

  @property
  def loss_type(self) -> spec.LossType:
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self) -> int:
    return 350343

  @property
  def num_eval_train_examples(self) -> int:
    return 43793

  @property
  def num_validation_examples(self) -> int:
    return 43793

  @property
  def num_test_examples(self) -> int:
    return 43793

  @property
  def eval_batch_size(self) -> int:
    return 32768
    # return 16384
  

  @property
  def train_mean(self):
    raise NotImplementedError

  @property
  def train_stddev(self):
    raise NotImplementedError

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 12_011  # ~3.3 hours

  @property
  def eval_period_time_sec(self) -> int:
    return 4 * 60

  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int):
    dataset_iter = input_pipeline.get_dataset_iter(split,
                                                   data_rng,
                                                   data_dir,
                                                   global_batch_size)
    if split != 'train':
      # Note that this stores the entire val dataset in memory.
      dataset_iter = itertools.cycle(dataset_iter)
    return dataset_iter

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,  # Dense or one-hot labels.
      logits_batch: spec.Tensor,
      mask_batch: Optional[spec.Tensor] = None,
      label_smoothing: float = 0.0) -> Dict[str, spec.Tensor]:  # differentiable
    """Evaluate the (masked) loss function at (label_batch, logits_batch).

    Return {'summed': scalar summed loss, 'n_valid_examples': scalar number of
    valid examples in batch, 'per_example': 1-d array of per-example losses}
    (not synced across devices).
    """
    print('in loss_fn')
    per_example_losses = self._binary_cross_entropy_with_mask(
        labels=label_batch,
        logits=logits_batch,
        mask=mask_batch,
        label_smoothing=label_smoothing)
    print('before mask')
    if mask_batch is not None:
      n_valid_examples = mask_batch.sum()
    else:
      n_valid_examples = len(per_example_losses)
    print('summing loss')
    summed_loss = per_example_losses.sum()
    print('returning summed loss')
    return {
        'summed': summed_loss,
        'n_valid_examples': n_valid_examples,
        'per_example': per_example_losses,
    }

  @property
  def step_hint(self) -> int:
    """Approx. steps the baseline can do in the allowed runtime budget."""
    return 52_000

  @abc.abstractmethod
  def _normalize_eval_metrics(
      self, num_examples: int, total_metrics: Dict[str,
                                                   Any]) -> Dict[str, float]:
    """Normalize eval metrics."""

  def _eval_batch(self,
                  params: spec.ParameterContainer,
                  batch: Dict[str, spec.Tensor],
                  model_state: spec.ModelAuxiliaryState,
                  rng: spec.RandomState) -> metrics.EvalMetrics:
    logits, _ = self.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.EVAL,
        rng,
        update_batch_norm=False)
    return self._eval_metric(batch['targets'], logits, batch['weights'])

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
    # del global_step
    data_rng, model_rng = prng.split(rng, 2)
    if split not in self._eval_iters:
      self._eval_iters[split] = self._build_input_queue(
          data_rng, split, data_dir, global_batch_size=global_batch_size)

    total_metrics = None
    num_eval_steps = int(math.ceil(float(num_examples) / global_batch_size))
    # Loop over graph batches in eval dataset.
    live_arrays = {}
    for s in range(num_eval_steps):      
      print(f'Eval step {s}')
      batch = next(self._eval_iters[split])
      jax.profiler.save_device_memory_profile(f"/logs/memory_filtered_{global_step}_{s}.prof")

      # batch_metrics = self._eval_batch(params, batch, model_state, model_rng)
      # print(f'merging metrics')
      # total_metrics = (
      #     batch_metrics
      #     if total_metrics is None else total_metrics.merge(batch_metrics))
      
      # len_arrays = len(jax.live_arrays())
      # print(f'Length of live arrays {len_arrays}')
      # live_arrays_new = { id(arr) : arr for arr in jax.live_arrays() }
      # diff_keys = live_arrays_new.keys() - live_arrays.keys()
      # diff_bytes = sum(map(attrgetter("nbytes"), [live_arrays_new[k] for k in diff_keys]))
      # live_arrays = live_arrays_new
      # if s == 1:
      #   print('DIFF IN ARRAYS:')
      #   for k in diff_keys:
      #     print(live_arrays_new[k].shape)
      #   print('DIFF in bytes')
      #   print(diff_bytes)
      # batch_shape = jax.tree.map(lambda x: x.shape, batch)
      # print('BATCH SHAPE')
      # print(batch_shape)
      # del(batch)
    if total_metrics is None:
      return {}
    # print('Normalizing eval metrics')
    # return self._normalize_eval_metrics(num_examples, total_metrics)
    return None
