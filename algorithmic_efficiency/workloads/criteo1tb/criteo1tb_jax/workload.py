"""Criteo1TB DLRM-Small workload implemented in Jax."""
import functools
import math
from typing import Dict, Optional, Tuple

from clu import metrics as clu_metrics
import flax
from flax import jax_utils
import jax
import jax.numpy as jnp

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.criteo1tb import input_pipeline
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_jax import dlrm_small_model
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_jax import metrics


_NUM_DENSE_FEATURES = 13
_VOCAB_SIZES = [1024 * 128] * 26


# We use CLU metrics to handle aggregating per-example outputs across batches so
# we can compute AUC metrics.
@flax.struct.dataclass
class EvalMetrics(clu_metrics.Collection):
  loss: clu_metrics.Average.from_output("loss")
  average_precision: metrics.BinaryMeanAveragePrecision
  auc_roc: metrics.BinaryAUCROC


class Criteo1TbDlrmSmallWorkload(spec.Workload):
  """Criteo1TB DLRM-Small Jax workload."""

  def __init__(self):
    self._eval_iters = {}
    self._param_shapes = None
    self._param_types = None
    self._flax_module = dlrm_small_model.DlrmSmall(
        vocab_sizes=_VOCAB_SIZES,
        total_vocab_sizes=sum(_VOCAB_SIZES),
        num_dense_features=_NUM_DENSE_FEATURES)

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['validation/auc_roc'] > self.target_value

  @property
  def target_value(self):
    return 0.8

  @property
  def loss_type(self):
    return spec.LossType.SIGMOID_CROSS_ENTROPY

  @property
  def num_train_examples(self):
    return 4_195_197_692

  @property
  def num_eval_train_examples(self):
    return 100_000

  @property
  def num_validation_examples(self):
    return 89_137_318 # DO NOT SUBMIT

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
    return 8 * 60 * 60  # 8 hours. # DO NOT SUBMIT

  @property
  def eval_period_time_sec(self):
    return 20 * 60  # 20 minutes. # DO NOT SUBMIT

  def build_input_queue(self,
                        data_rng: jax.random.PRNGKey,
                        split: str,
                        data_dir: str,
                        global_batch_size: int,
                        num_batches: Optional[int] = None,
                        repeat_final_dataset: bool = False):
    del data_rng
    ds = input_pipeline.get_criteo1tb_dataset(
        split=split,
        data_dir=data_dir,
        is_training=(split =='train'),
        global_batch_size=global_batch_size,
        num_dense_features=_NUM_DENSE_FEATURES,
        vocab_sizes=_VOCAB_SIZES,
        num_batches=num_batches,
        repeat_final_dataset=repeat_final_dataset)
    for batch in iter(ds):
      batch = jax.tree_map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access
      yield batch

  @functools.partial(
      jax.pmap, axis_name='batch', static_broadcasted_argnums=(0,))
  def eval_step_pmapped(self, params, batch):
    """Calculate evaluation metrics on a batch."""
    inputs = batch['inputs']
    targets = batch['targets']
    weights = batch['weights']
    logits = self._flax_module.apply({'params': params}, inputs, targets)
    per_example_losses = metrics.per_example_sigmoid_binary_cross_entropy(
        logits, targets)
    return EvalMetrics.gather_from_model_output(
        loss=per_example_losses, targets=targets, logits=logits, mask=weights)

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str) -> Dict[str, float]:
    """Run a full evaluation of the model."""
    del model_state
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
    metrics_bundle = EvalMetrics.empty()
    for _ in range(num_batches):
      eval_batch = next(self._eval_iters[split])
      metrics_bundle = metrics_bundle.merge(
          self.eval_step_pmapped(params, eval_batch).unreplicate())
    return metrics_bundle.compute()

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
    per_example_losses = metrics.per_example_sigmoid_binary_cross_entropy(
        logits=logits_batch, targets=label_batch)
    if mask_batch is not None:
      weighted_losses = per_example_losses * mask_batch
      normalization = mask_batch.sum()
    else:
      weighted_losses = per_example_losses
    normalization = label_batch.shape[0]
    return jnp.sum(weighted_losses, axis=-1) / normalization

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    rng, init_rng = jax.random.split(rng)
    init_fake_batch_size = 2
    input_size = _NUM_DENSE_FEATURES + len(_VOCAB_SIZES)
    input_shape = (init_fake_batch_size, input_size)
    target_shape = (init_fake_batch_size, input_size)

    initial_variables = jax.jit(self._flax_module.init)(
        init_rng,
        jnp.ones(input_shape, jnp.float32),
        jnp.ones(target_shape, jnp.float32))

    initial_params = initial_variables['params']
    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      initial_params)
    return jax_utils.replicate(initial_params), None

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del mode
    del rng
    del update_batch_norm
    inputs = augmented_and_preprocessed_input_batch['inputs']
    targets = augmented_and_preprocessed_input_batch['targets']
    logits_batch = self._flax_module.apply({'params': params}, inputs, targets)
    return logits_batch, None

  @property
  def step_hint(self):
    return 64_000 # DO NOT SUBMIT
