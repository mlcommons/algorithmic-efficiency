import functools
import math
from typing import Dict, Optional, Tuple

from flax import jax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_jax import \
    models
from algorithmic_efficiency.workloads.librispeech_deepspeech.workload import \
    BaseDeepspeechLibrispeechWorkload


class LibriSpeechDeepSpeechWorkload(BaseDeepspeechLibrispeechWorkload):

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """Deepspeech model init function.

    Here we use dropout_rate as feed_forward_dropout_rate, and aux_dropout_rate
    as input_dropout_rate.
    """
    model_config = models.DeepspeechConfig(
        feed_forward_dropout_rate=dropout_rate,
        use_specaug=self.use_specaug,
        input_dropout_rate=aux_dropout_rate)
    self._model = models.Deepspeech(model_config)
    input_shape = [(320000,), (320000,)]
    fake_input_batch = [np.zeros((2, *x), jnp.float32) for x in input_shape]

    model_init_fn = jax.jit(functools.partial(self._model.init, train=False))

    params_rng, dropout_rng = jax.random.split(rng, 2)
    variables = model_init_fn({'params': params_rng, 'dropout': dropout_rng},
                              *fake_input_batch)

    model_state = variables['batch_stats']
    params = variables['params']
    self._param_shapes = param_utils.jax_param_shapes(params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
    model_state = jax_utils.replicate(model_state)
    params = jax_utils.replicate(params)
    return params, model_state

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key == 'Dense_0'

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    variables = {'params': params, **model_state}
    inputs, input_paddings = augmented_and_preprocessed_input_batch['inputs']
    is_train_mode = mode == spec.ForwardPassMode.TRAIN
    if update_batch_norm or is_train_mode:
      (logits, logit_paddings), new_model_state = self._model.apply(
          variables,
          inputs,
          input_paddings,
          train=True,
          rngs={'dropout' : rng},
          mutable=['batch_stats'])
      return (logits, logit_paddings), new_model_state
    else:
      logits, logit_paddings = self._model.apply(
          variables,
          inputs,
          input_paddings,
          train=False,
          mutable=False)
      return (logits, logit_paddings), model_state

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: Tuple[spec.Tensor, spec.Tensor],  # (label_batch, padding)
      logits_batch: Tuple[spec.Tensor, spec.Tensor],  # (logits_batch, padding)
      mask_batch: Optional[spec.Tensor] = None,
      label_smoothing: float = 0.0) -> Dict[str, spec.Tensor]:  # differentiable
    """Evaluate the (masked) loss function at (label_batch, logits_batch).

    Return {'summed': scalar summed loss, 'n_valid_examples': scalar number of
    valid examples in batch, 'per_example': 1-d array of per-example losses}
    (not synced across devices).
    """
    del label_smoothing
    logits, logit_paddings = logits_batch
    targets, target_paddings = label_batch
    logprobs = nn.log_softmax(logits)
    per_example_losses = self.ctc_loss(logprobs,
                                       logit_paddings,
                                       targets,
                                       target_paddings)
    # mask_batch is assumed to be shape [batch].
    if mask_batch is not None:
      per_example_losses *= mask_batch
      mask_batch = jnp.logical_and(mask_batch, 1 - target_paddings)
    else:
      mask_batch = 1 - target_paddings
    n_valid_examples = jnp.maximum(mask_batch.sum(), 1)
    summed_loss = per_example_losses.sum()
    return {
        'summed': summed_loss,
        'n_valid_examples': n_valid_examples,
        'per_example': per_example_losses,
    }

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
    del global_step
    if model_state is not None:
      # Sync batch statistics across replicas before evaluating.
      model_state = self.sync_batch_stats(model_state)

    num_batches = int(math.ceil(num_examples / global_batch_size))
    if split not in self._eval_iters:
      self._eval_iters[split] = self._build_input_queue(
          rng, split, data_dir, global_batch_size, num_batches=num_batches)

    metrics_report = None
    for _ in range(num_batches):
      eval_batch = next(self._eval_iters[split])
      computed_metrics = self.eval_step_pmapped(params,
                                                eval_batch,
                                                model_state,
                                                rng).unreplicate()

      if metrics_report is None:
        metrics_report = computed_metrics
      else:
        # `merge` aggregates the metrics across batches.
        metrics_report = metrics_report.merge(computed_metrics)

    computed_metrics = metrics_report.compute()

    return computed_metrics
