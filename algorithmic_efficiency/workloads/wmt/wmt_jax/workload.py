"""WMT workload implemented in Jax."""
import collections
import functools
from typing import Dict, Optional, Tuple

from absl import logging
from flax import linen as nn
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.wmt import bleu
from algorithmic_efficiency.workloads.wmt import decode
from algorithmic_efficiency.workloads.wmt.wmt_jax import models
from algorithmic_efficiency.workloads.wmt.workload import BaseWmtWorkload


def _pad_examples(desired_batch_size, x):
  """Expand batch to desired size by repeating last slice."""
  batch_pad = desired_batch_size - x.shape[0]
  return np.concatenate([x, np.tile(x[-1], (batch_pad, 1))], axis=0)


def _per_host_sum_pmap(in_tree):
  """Execute psum on in_tree"s leaves over one device per host."""
  host2devices = collections.defaultdict(list)
  for d in jax.devices():
    host2devices[d.host_id].append(d)
  devices = [host2devices[k][0] for k in host2devices]
  host_psum = jax.pmap(lambda x: jax.lax.psum(x, "i"), "i", devices=devices)

  def pre_pmap(xs):
    return jax.tree_map(lambda x: jnp.broadcast_to(x, (1,) + x.shape), xs)

  def post_pmap(xs):
    return jax.tree_map(lambda x: x[0], xs)

  return post_pmap(host_psum(pre_pmap(in_tree)))


def _to_host(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


class WmtWorkload(BaseWmtWorkload):
  """WMT Jax workload."""

  def __init__(self):
    self._initial_params = None
    self._param_types = None
    self._tokenizer = None
    self._vocab_size = 32000
    self._train_config = models.TransformerConfig()
    self._eval_config = models.TransformerConfig(deterministic=True)

  def compute_weighted_cross_entropy(self,
                                     logits,
                                     targets,
                                     weights,
                                     label_smoothing=0.0):
    """Compute weighted cross entropy and entropy for log probs and targets.

    Args:
     logits: [batch, length, num_classes] float array.
     targets: categorical targets [batch, length] int array.
     weights: array of shape [batch, length].
     label_smoothing: label smoothing constant, used to determine the on and off
       values.

    Returns:
      Tuple of loss for every example and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
      raise ValueError(
          f"Incorrect shapes. Got shape {str(logits.shape)} logits "
          f"and {str(targets.shape)} targets")
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (self._vocab_size - 1)
    normalizing_constant = -(
        confidence * jnp.log(confidence) +
        ((self._vocab_size - 1) * low_confidence *
         jnp.log(low_confidence + 1e-20)))
    soft_targets = common_utils.onehot(
        targets, self._vocab_size, on_value=confidence, off_value=low_confidence)

    loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
    loss = loss - normalizing_constant

    normalizing_factor = np.prod(targets.shape)
    if weights is not None:
      loss = loss * weights
      normalizing_factor = weights.sum()

    return loss, normalizing_factor

  def compute_weighted_accuracy(self, logits, targets, weights=None):
    """Compute weighted accuracy for log probs and targets.

    Args:
     logits: [batch, length, num_classes] float array.
     targets: categorical targets [batch, length] int array.
     weights: None or array of shape [batch, length]

    Returns:
      Accuracy summed across the batch.
    """
    if logits.ndim != targets.ndim + 1:
      raise ValueError(f"Incorrect shapes. Got shape {str(logits.shape)} logits"
                       f" and {str(targets.shape)} targets")
    accuracy = jnp.equal(jnp.argmax(logits, axis=-1), targets)
    if weights is not None:
      accuracy = accuracy * weights
    return accuracy.sum()

  @functools.partial(
      jax.pmap,
      axis_name='batch',
      static_broadcasted_argnums=(0,))
  def eval_step(self, params, batch):
    """Calculate evaluation metrics on a batch."""
    inputs, targets = batch["inputs"], batch["targets"]
    weights = jnp.where(targets > 0, 1.0, 0.0)
    logits = models.Transformer(self._eval_config).apply(
        {"params": params}, inputs, targets)
    loss, weight_sum = self.compute_weighted_cross_entropy(
        logits, targets, weights, 0.0)
    metrics = {
        "loss": loss.sum(),
        "accuracy": self.compute_weighted_accuracy(logits, targets, weights),
        "denominator": weight_sum,
    }
    metrics = jax.lax.psum(metrics, axis_name="batch")
    return metrics

  @functools.partial(
      jax.pmap,
      axis_name='batch',
      static_broadcasted_argnums=(0,))
  def initialize_cache(self, inputs, max_decode_len=256):
    """Initialize a cache for a given input shape and max decode length."""
    config = models.TransformerConfig(deterministic=True, decode=True)
    target_shape = (inputs.shape[0], max_decode_len) + inputs.shape[2:]
    initial_variables = models.Transformer(config).init(
        jax.random.PRNGKey(0),
        jnp.ones(inputs.shape, jnp.float32),
        jnp.ones(target_shape, jnp.float32))
    return initial_variables["cache"]

  # eos_id, max_decode_len are constant.
  @functools.partial(
      jax.pmap,
      axis_name='batch',
      static_broadcasted_argnums=(0, 4, 5))
  def predict_step(self,
                   inputs,
                   params,
                   cache,
                   eos_id,
                   max_decode_len,
                   beam_size=4):
    """Predict translation with fast decoding beam search on a batch."""
    config = models.TransformerConfig(deterministic=True, decode=True)
    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * beam_size, where each batch item"s data is expanded in-place
    # rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
    encoded_inputs = decode.flat_batch_beam_expand(
        models.Transformer(config).apply({"params": params},
                                         inputs,
                                         method=models.Transformer.encode),
        beam_size)
    raw_inputs = decode.flat_batch_beam_expand(inputs, beam_size)

    def tokens_ids_to_logits(flat_ids, flat_cache):
      """Token slice to logits from decoder model."""
      # --> [batch * beam, 1, vocab]
      flat_logits, new_vars = models.Transformer(config).apply(
          {
              "params": params,
              "cache": flat_cache
          },
          encoded_inputs,
          raw_inputs,  # only needed for input padding mask
          flat_ids,
          mutable=["cache"],
          method=models.Transformer.decode)
      new_flat_cache = new_vars["cache"]
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
                                   params,
                                   target,
                                   predict_ds: tf.data.Dataset,
                                   max_predict_length: int,
                                   num_eval_steps: int):
    """Translates the `predict_ds` and calculates the BLEU score."""
    n_devices = jax.local_device_count()
    logging.info("Translating evaluation dataset.")
    sources, references, predictions = [], [], []
    for _, pred_batch in zip(range(num_eval_steps + 1), predict_ds):
      pred_batch = jax.tree_map(lambda x: x._numpy(), pred_batch)  # pylint: disable=protected-access
      # Handle final odd-sized batch by padding instead of dropping it.
      cur_pred_batch_size = pred_batch["inputs"].shape[0]
      if cur_pred_batch_size % n_devices:
        padded_size = int(np.ceil(cur_pred_batch_size / n_devices) * n_devices)
        pred_batch = jax.tree_map(
            functools.partial(_pad_examples, padded_size),  # pylint: disable=cell-var-from-loop
            pred_batch)
      cache = self.initialize_cache(pred_batch["inputs"])
      predicted = self.predict_step(
          pred_batch["inputs"],
          params,
          target,
          cache,
          decode.EOS_ID,
          max_predict_length)
      predicted = _to_host(predicted)
      inputs = _to_host(pred_batch["inputs"])
      targets = _to_host(pred_batch["targets"])
      # Iterate through non-padding examples of batch.
      for i, s in enumerate(predicted[:cur_pred_batch_size]):
        sources.append(self._decode_tokens(inputs[i]))
        references.append(self._decode_tokens(targets[i]))
        predictions.append(self._decode_tokens(s))

    # Calculate BLEU score for translated eval corpus against reference.
    bleu_matches = bleu.bleu_partial(references, predictions)
    all_bleu_matches = _per_host_sum_pmap(bleu_matches)
    bleu_score = bleu.complete_bleu(*all_bleu_matches)
    return bleu_score

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    rng, init_rng = jax.random.split(rng)
    init_fake_batch_size = 2
    input_shape = (init_fake_batch_size, 256)
    target_shape = (init_fake_batch_size, 256)

    initial_variables = jax.jit(models.Transformer(self._eval_config).init)(
        init_rng,
        jnp.ones(input_shape, jnp.float32),
        jnp.ones(target_shape, jnp.float32))

    initial_params = initial_variables["params"]
    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      initial_params)
    return initial_params, None

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: spec.Tensor,
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del rng
    del update_batch_norm

    if mode == spec.ForwardPassMode.TRAIN:
      model_config = self._train_config
    else:
      model_config = self._eval_config
    inputs = augmented_and_preprocessed_input_batch["inputs"]
    targets = augmented_and_preprocessed_input_batch["targets"]
    logits_batch = models.Transformer(model_config).apply({"params": params},
                                                          inputs,
                                                          targets)
    return logits_batch, None

  def loss_fn(
      self,
      label_batch: spec.Tensor,  # Dense (not one-hot) labels.
      logits_batch: spec.Tensor,
      mask_batch: Optional[spec.Tensor]) -> spec.Tensor:
    del mask_batch
    weights = jnp.where(label_batch > 0, 1.0, 0.0)
    loss = self.compute_weighted_cross_entropy(logits_batch,
                                               label_batch,
                                               weights)
    return loss
