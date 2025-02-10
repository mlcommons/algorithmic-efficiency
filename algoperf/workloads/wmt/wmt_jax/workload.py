"""WMT workload implemented in Jax."""

from dataclasses import replace
import functools
from typing import Any, Dict, Iterator, Optional, Tuple

from absl import logging
from flax import jax_utils
from flax import linen as nn
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax

from algoperf import param_utils
from algoperf import spec
from algoperf.workloads.wmt import bleu
from algoperf.workloads.wmt.wmt_jax import decode
from algoperf.workloads.wmt.wmt_jax import models
from algoperf.workloads.wmt.workload import BaseWmtWorkload


def _to_host(x: spec.Tensor) -> spec.Tensor:
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


class WmtWorkload(BaseWmtWorkload):
  """WMT Jax workload."""

  def compute_weighted_cross_entropy(
      self,
      logits: spec.Tensor,
      targets: spec.Tensor,
      weights: Optional[spec.Tensor] = None,
      label_smoothing: float = 0.1) -> Dict[str, spec.Tensor]:  # differentiable
    """Compute weighted cross entropy and entropy for log probs and targets.

    Args:
     logits: [batch, length, num_classes] float array.
     targets: categorical targets [batch, length] int array.
     weights: array of shape [batch, length].
     label_smoothing: label smoothing constant, used to determine the on and off
       values.

    Returns:
      {'summed': scalar summed loss, 'n_valid_examples': scalar number of
      valid examples in batch, 'per_example': 1-d array of per-example losses}
    """
    if logits.ndim != targets.ndim + 1:
      raise ValueError(f'Incorrect shapes. Got shape {logits.shape} logits and '
                       f'{targets.shape} targets.')
    smoothed_targets = optax.smooth_labels(
        common_utils.onehot(targets, self._vocab_size), label_smoothing)

    per_example_losses = -jnp.sum(
        smoothed_targets * nn.log_softmax(logits), axis=-1)
    if weights is None:
      weights = jnp.ones_like(targets)
    per_example_losses = jnp.where(weights, per_example_losses, 0.)
    summed_loss = per_example_losses.sum()
    n_valid_examples = weights.sum()
    return {
        'summed': summed_loss,
        'n_valid_examples': n_valid_examples,
        'per_example': per_example_losses,
    }

  @functools.partial(
      jax.pmap, axis_name='batch', static_broadcasted_argnums=(0,))
  def eval_step_pmapped(
      self, params: spec.ParameterContainer,
      batch: Dict[str, spec.Tensor]) -> Dict[str, spec.Tensor]:
    """Calculate evaluation metrics on a batch."""
    inputs = batch['inputs']
    targets = batch['targets']
    weights = batch['weights']
    logits = self._eval_model.apply({'params': params}, inputs, targets)
    summed_loss = self.compute_weighted_cross_entropy(logits,
                                                      targets,
                                                      weights,
                                                      0.0)['summed']
    acc_sum, weight_sum = self.compute_weighted_accuracy(
        logits, targets, weights)
    return {
        'loss': summed_loss,
        'accuracy': acc_sum,
        'denominator': weight_sum,
    }

  def eval_step(self,
                params: spec.ParameterContainer,
                batch: Dict[str, spec.Tensor]) -> Dict[str, spec.Tensor]:
    replicated_eval_metrics = self.eval_step_pmapped(params, batch)
    return jax.tree_map(lambda x: jnp.sum(x, axis=0), replicated_eval_metrics)

  @functools.partial(
      jax.pmap, axis_name='batch', static_broadcasted_argnums=(0,))
  def initialize_cache(self,
                       inputs: spec.Tensor,
                       max_decode_len: int = 256) -> Dict[str, spec.Tensor]:
    """Initialize a cache for a given input shape and max decode length."""
    config = models.TransformerConfig(deterministic=True, decode=True)
    target_shape = (inputs.shape[0], max_decode_len) + inputs.shape[2:]
    initial_variables = models.Transformer(config).init(
        jax.random.PRNGKey(0),
        jnp.ones(inputs.shape, jnp.float32),
        jnp.ones(target_shape, jnp.float32))
    return initial_variables['cache']

  # eos_id, max_decode_len are constant.
  @functools.partial(
      jax.pmap, axis_name='batch', static_broadcasted_argnums=(0, 4, 5))
  def predict_step(self,
                   inputs: spec.Tensor,
                   params: spec.ParameterContainer,
                   cache: Dict[str, spec.Tensor],
                   eos_id: int,
                   max_decode_len: int,
                   beam_size: int = 4) -> spec.Tensor:
    """Predict translation with fast decoding beam search on a batch."""
    config = replace(self._eval_model.config, decode=True)
    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * beam_size, where each batch item's data is expanded in-place
    # rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
    encoded_inputs = decode.flat_batch_beam_expand(
        models.Transformer(config).apply({'params': params},
                                         inputs,
                                         method=models.Transformer.encode),
        beam_size)
    raw_inputs = decode.flat_batch_beam_expand(inputs, beam_size)

    def tokens_ids_to_logits(
        flat_ids: spec.Tensor, flat_cache: Dict[str, spec.Tensor]
    ) -> Tuple[spec.Tensor, Dict[str, spec.Tensor]]:
      """Token slice to logits from decoder model."""
      # --> [batch * beam, 1, vocab]
      flat_logits, new_vars = models.Transformer(config).apply(
          {
              'params': params,
              'cache': flat_cache,
          },
          encoded_inputs,
          raw_inputs,  # only needed for input padding mask
          flat_ids,
          mutable=['cache'],
          method=models.Transformer.decode)
      new_flat_cache = new_vars['cache']
      # Remove singleton sequence-length dimension:
      # [batch * beam, 1, vocab] --> [batch * beam, vocab]
      flat_logits = flat_logits.squeeze(axis=1)
      return flat_logits, new_flat_cache

    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    beam_seqs, _ = decode.beam_search(
        inputs,
        cache,
        tokens_ids_to_logits,
        beam_size=beam_size,
        alpha=0.6,
        eos_id=eos_id,
        max_decode_len=max_decode_len)

    # Beam search returns [n_batch, n_beam, n_length + 1] with beam dimension
    # sorted in increasing order of log-probability.
    # Return the highest scoring beam sequence, drop first dummy 0 token.
    return beam_seqs[:, -1, 1:]

  def translate_and_calculate_bleu(self,
                                   params: spec.ParameterContainer,
                                   ds_iter: Iterator,
                                   num_batches: int,
                                   max_predict_length: int) -> spec.Tensor:
    """Translates the `predict_ds` and calculates the BLEU score."""
    logging.info('Translating evaluation dataset.')
    references, predictions = [], []
    for _ in range(num_batches):
      pred_batch = next(ds_iter)
      cache = self.initialize_cache(pred_batch['inputs'])
      predicted = self.predict_step(pred_batch['inputs'],
                                    params,
                                    cache,
                                    decode.EOS_ID,
                                    max_predict_length)
      predicted = _to_host(predicted)
      targets = _to_host(pred_batch['targets'])
      # Find actual batch size, ignoring the potential padding.
      weights = pred_batch.get('weights')
      if weights is not None:
        weights = _to_host(weights)
        actual_batch_size = int(weights.sum(0)[0].item())
      else:
        actual_batch_size = len(predicted)
      # Iterate through non-padding examples of batch.
      for idx in range(actual_batch_size):
        references.append(self._decode_tokens(targets[idx]))
        predictions.append(self._decode_tokens(predicted[idx]))

    # Calculate BLEU score for translated eval corpus against reference.
    bleu_score = bleu.corpus_bleu(predictions, [references]).score
    return bleu_score

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """aux_dropout_rate is used as attention_dropout_rate."""

    init_fake_batch_size = 2
    input_shape = (init_fake_batch_size, 256)
    target_shape = (init_fake_batch_size, 256)

    if self.activation == 'relu':
      activation = nn.relu
    elif self.activation == 'tanh':
      activation = jnp.tanh
    else:
      raise ValueError(f'Unknown activation function {self.activation}.')

    model_config = models.TransformerConfig(
        dropout_rate=dropout_rate,
        attention_dropout_rate=aux_dropout_rate,
        pre_ln=self.pre_ln,
        attention_temp=self.attention_temp,
        activation=activation,
        glu=self.glu)
    self._train_model = models.Transformer(model_config)
    eval_config = replace(model_config, deterministic=True)
    self._eval_model = models.Transformer(eval_config)
    params_rng, dropout_rng = jax.random.split(rng)
    initial_variables = jax.jit(
        self._eval_model.init)({'params': params_rng, 'dropout': dropout_rng},
                               jnp.ones(input_shape, jnp.float32),
                               jnp.ones(target_shape, jnp.float32))

    initial_params = initial_variables['params']
    self._param_shapes = param_utils.jax_param_shapes(initial_params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
    return jax_utils.replicate(initial_params), None

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key == 'shared_embedding'

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del update_batch_norm

    inputs = augmented_and_preprocessed_input_batch.get('inputs', None)
    targets = augmented_and_preprocessed_input_batch.get('targets', None)
    inputs_positions = augmented_and_preprocessed_input_batch.get(
        'inputs_position', None)
    targets_positions = augmented_and_preprocessed_input_batch.get(
        'targets_position', None)
    inputs_segmentations = augmented_and_preprocessed_input_batch.get(
        'inputs_segmentation', None)
    targets_segmentations = augmented_and_preprocessed_input_batch.get(
        'targets_segmentation', None)

    if mode == spec.ForwardPassMode.TRAIN:
      model = self._train_model
    else:
      model = self._eval_model

    logits_batch = model.apply({'params': params},
                               inputs,
                               targets,
                               inputs_positions=inputs_positions,
                               targets_positions=targets_positions,
                               inputs_segmentation=inputs_segmentations,
                               targets_segmentation=targets_segmentations,
                               rngs={'dropout': rng})
    return logits_batch, None

  def _normalize_eval_metrics(
      self, num_examples: int, total_metrics: Dict[str,
                                                   Any]) -> Dict[str, float]:
    """Normalize eval metrics."""
    del num_examples
    eval_denominator = total_metrics.pop('denominator')
    return jax.tree_map(lambda x: float(x / eval_denominator), total_metrics)


class WmtWorkloadPostLN(WmtWorkload):
  """WMT Jax workload with post instead of pre layer norm."""

  @property
  def validation_target_value(self) -> float:
    return 30.0779

  @property
  def test_target_value(self) -> float:
    return 29.8982

  @property
  def pre_ln(self) -> bool:
    return False


class WmtWorkloadAttentionTemp(WmtWorkload):
  """WMT Jax workload with attention temperature = 4.0."""

  @property
  def validation_target_value(self) -> float:
    return 29.3379

  @property
  def test_target_value(self) -> float:
    return 29.4143

  @property
  def attention_temp(self) -> float:
    return 4.0


class WmtWorkloadGLUTanH(WmtWorkload):
  """WMT Jax workload with GLU and TanH activations."""

  @property
  def validation_target_value(self) -> float:
    return 29.5779

  @property
  def test_target_value(self) -> float:
    return 29.0515

  @property
  def activation(self) -> str:
    return 'tanh'

  @property
  def glu(self) -> bool:
    return True
