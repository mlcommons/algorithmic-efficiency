"""WMT workload implemented in Jax."""
import collections
import functools
from typing import Tuple

from . import bleu
from . import decode
from . import input_pipeline
from . import models
from absl import logging
from flax import linen as nn
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import spec
import tensorflow as tf

VOCAB_PATH = "./wmt_256/sentencepiece_model"
WORKDIR = "./wmt_256"


class WMTWorkload(spec.Workload):
  """A WMT workload."""

  def __init__(self):
    self._eval_ds = None
    self._train_ds = None
    self._predict_ds = None
    self._encoder = None
    self._vocab_size = 32000
    self._per_device_batch_size = None
    self._train_config = None
    self._eval_config = None
    self._predict_config = None
    self._p_eval_step = None
    self._p_init_cache = None
    self._p_pred_step = None

  def compute_weighted_cross_entropy(self,
                                     logits,
                                     targets,
                                     weights=None,
                                     label_smoothing=0.0):
    """Compute weighted cross entropy and entropy for log probs and targets.

    Args:
     logits: [batch, length, num_classes] float array.
     targets: categorical targets [batch, length] int array.
     weights: None or array of shape [batch, length].
     label_smoothing: label smoothing constant, used to determine the on and off
       values.

    Returns:
      Tuple of loss for every example and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
      raise ValueError("Incorrect shapes. Got shape %s logits and %s targets" %
                       (str(logits.shape), str(targets.shape)))
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
        confidence * jnp.log(confidence) +
        (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
    soft_targets = common_utils.onehot(
        targets, vocab_size, on_value=confidence, off_value=low_confidence)

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
      Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
      raise ValueError("Incorrect shapes. Got shape %s logits and %s targets" %
                       (str(logits.shape), str(targets.shape)))
    loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
    normalizing_factor = np.prod(logits.shape[:-1])
    if weights is not None:
      loss = loss * weights
      normalizing_factor = weights.sum()

    return loss.sum(), normalizing_factor

  def compute_metrics(self, logits, labels, weights):
    """Compute summary metrics."""
    loss, weight_sum = self.compute_weighted_cross_entropy(
        logits, labels, weights, 0.0)
    acc, _ = self.compute_weighted_accuracy(logits, labels, weights)
    metrics = {
        "loss": loss.sum(),
        "accuracy": acc,
        "denominator": weight_sum,
    }
    metrics = jax.lax.psum(metrics, axis_name="batch")
    return metrics

  # Primary eval / decode step functions.
  # -----------------------------------------------------------------------------

  def eval_step(self, params, batch, config):
    """Calculate evaluation metrics on a batch."""
    inputs, targets = batch["inputs"], batch["targets"]
    weights = jnp.where(targets > 0, 1.0, 0.0)
    logits = models.Transformer(config).apply({"params": params}, inputs,
                                              targets)

    return self.compute_metrics(logits, targets, weights)

  def initialize_cache(self, inputs, max_decode_len, config):
    """Initialize a cache for a given input shape and max decode length."""
    target_shape = (inputs.shape[0], max_decode_len) + inputs.shape[2:]
    initial_variables = models.Transformer(config).init(
        jax.random.PRNGKey(0), jnp.ones(inputs.shape, jnp.float32),
        jnp.ones(target_shape, jnp.float32))
    return initial_variables["cache"]

  def predict_step(self,
                   inputs,
                   params,
                   cache,
                   eos_id,
                   max_decode_len,
                   config,
                   beam_size=4):
    """Predict translation with fast decoding beam search on a batch."""
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

  # Utils for prediction and BLEU calculation
  # -----------------------------------------------------------------------------

  def pad_examples(self, x, desired_batch_size):
    """Expand batch to desired size by repeating last slice."""
    batch_pad = desired_batch_size - x.shape[0]
    return np.concatenate([x, np.tile(x[-1], (batch_pad, 1))], axis=0)

  def per_host_sum_pmap(self, in_tree):
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

  def tohost(self, x):
    """Collect batches from all devices to host and flatten batch dimensions."""
    n_device, n_batch, *remaining_dims = x.shape
    return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))

  def evaluate(self, p_eval_step, target, eval_ds: tf.data.Dataset,
               num_eval_steps: int):
    """Evaluate the target an return a dictionary with the metrics."""
    logging.info("Gathering evaluation metrics.")
    eval_metrics = []
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    for _, eval_batch in zip(range(num_eval_steps), eval_iter):
      eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
      eval_batch = common_utils.shard(eval_batch)
      metrics = p_eval_step(target, eval_batch)
      eval_metrics.append(metrics)
    eval_metrics = common_utils.get_metrics(eval_metrics)
    eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
    eval_denominator = eval_metrics_sums.pop("denominator")
    eval_summary = jax.tree_map(
        lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
        eval_metrics_sums)
    return eval_summary

  def translate_and_calculate_bleu(self, p_pred_step, p_init_cache, target,
                                   predict_ds: tf.data.Dataset, decode_tokens,
                                   max_predict_length: int):
    """Translates the `predict_ds` and calculates the BLEU score."""
    n_devices = jax.local_device_count()
    logging.info("Translating evaluation dataset.")
    sources, references, predictions = [], [], []
    for pred_batch in predict_ds:
      pred_batch = jax.tree_map(lambda x: x._numpy(), pred_batch)  # pylint: disable=protected-access
      # Handle final odd-sized batch by padding instead of dropping it.
      cur_pred_batch_size = pred_batch["inputs"].shape[0]
      if cur_pred_batch_size % n_devices:
        padded_size = int(np.ceil(cur_pred_batch_size / n_devices) * n_devices)
        pred_batch = jax.tree_map(
            lambda x: self.pad_examples(x, padded_size),  # pylint: disable=cell-var-from-loop
            pred_batch)
      pred_batch = common_utils.shard(pred_batch)
      cache = p_init_cache(pred_batch["inputs"])
      predicted = p_pred_step(pred_batch["inputs"], target, cache,
                              decode.EOS_ID, max_predict_length)
      predicted = self.tohost(predicted)
      inputs = self.tohost(pred_batch["inputs"])
      targets = self.tohost(pred_batch["targets"])
      # Iterate through non-padding examples of batch.
      for i, s in enumerate(predicted[:cur_pred_batch_size]):
        sources.append(decode_tokens(inputs[i]))
        references.append(decode_tokens(targets[i]))
        predictions.append(decode_tokens(s))
    logging.info("Translation: %d predictions %d references %d sources.",
                 len(predictions), len(references), len(sources))

    # Calculate BLEU score for translated eval corpus against reference.
    bleu_matches = bleu.bleu_partial(references, predictions)
    all_bleu_matches = self.per_host_sum_pmap(bleu_matches)
    bleu_score = bleu.complete_bleu(*all_bleu_matches)

    # Save translation samples for tensorboard.
    exemplars = ""
    for n in np.random.choice(np.arange(len(predictions)), 8):
      exemplars += f"{sources[n]}\n\n{references[n]}\n\n{predictions[n]}\n\n"
    return exemplars, bleu_score

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result["bleu"] > self.target_value

  def build_input_queue(self, data_rng: jax.random.PRNGKey, split: str,
                        data_dir: str, batch_size: int):
    tf.io.gfile.makedirs(WORKDIR)
    self._per_device_batch_size = batch_size
    self._train_ds, self._eval_ds, self._predict_ds, self._encoder = input_pipeline.get_wmt_datasets(
        vocab_size=self._vocab_size,
        batch_size=jax.local_device_count() * batch_size,
        reverse_translation=True,
        vocab_path=VOCAB_PATH)
    self._vocab_size = int(self._encoder.vocab_size())
    return iter(self._train_ds)

  @property
  def param_shapes(self):
    init_params, _ = self.init_model_fn(jax.random.PRNGKey(0))
    return jax.tree_map(lambda x: spec.ShapeTuple(x.shape), init_params)

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
  def num_eval_examples(self):
    return 3004

  @property
  def train_mean(self):
    return 0.0

  @property
  def train_stddev(self):
    return 1.0

  def model_params_types(self):
    pass

  @property
  def max_allowed_runtime_sec(self):
    return 80000

  @property
  def eval_period_time_sec(self):
    return 800

  def _decode_tokens(self, toks):
    valid_toks = toks[:np.argmax(toks == decode.EOS_ID) + 1].astype(np.int32)
    return self._encoder.detokenize(valid_toks).numpy().decode("utf-8")

  # Return whether or not a key in spec.ParameterContainer is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def preprocess_for_train(self, selected_raw_input_batch: spec.Tensor,
                           selected_label_batch: spec.Tensor,
                           train_mean: spec.Tensor, train_stddev: spec.Tensor,
                           rng: spec.RandomState) -> spec.Tensor:
    del train_mean
    del train_stddev
    del rng
    return selected_raw_input_batch, selected_label_batch

  def preprocess_for_eval(self, raw_input_batch: spec.Tensor,
                          train_mean: spec.Tensor,
                          train_stddev: spec.Tensor) -> spec.Tensor:
    del train_mean
    del train_stddev
    return raw_input_batch

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    self._train_config = models.TransformerConfig(
        vocab_size=self._vocab_size, output_vocab_size=self._vocab_size)
    self._eval_config = models.TransformerConfig(
        vocab_size=self._vocab_size,
        output_vocab_size=self._vocab_size,
        deterministic=True)
    self._predict_config = models.TransformerConfig(
        vocab_size=self._vocab_size,
        output_vocab_size=self._vocab_size,
        deterministic=True,
        decode=True)
    self._p_eval_step = jax.pmap(
        functools.partial(self.eval_step, config=self._eval_config),
        axis_name="batch")
    self._p_init_cache = jax.pmap(
        functools.partial(
            self.initialize_cache,
            max_decode_len=256,
            config=self._predict_config),
        axis_name="batch")
    self._p_pred_step = jax.pmap(
        functools.partial(
            self.predict_step, config=self._predict_config, beam_size=4),
        axis_name="batch",
        static_broadcasted_argnums=(3, 4))  # eos token, max_length are constant

    rng, init_rng = jax.random.split(rng)
    input_shape = (self._per_device_batch_size, 256)
    target_shape = (self._per_device_batch_size, 256)

    initial_variables = jax.jit(models.Transformer(self._eval_config).init)(
        init_rng, jnp.ones(input_shape, jnp.float32),
        jnp.ones(target_shape, jnp.float32))

    initial_params = initial_variables["params"]

    return initial_params, None

  def model_fn(
      self, params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: spec.Tensor,
      model_state: spec.ModelAuxiliaryState, mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del rng
    del update_batch_norm

    model_config = self._train_config if mode == spec.ForwardPassMode.TRAIN else self._eval_config
    inputs, targets = augmented_and_preprocessed_input_batch[
        "inputs"], augmented_and_preprocessed_input_batch["targets"]
    logits_batch = models.Transformer(model_config).apply({"params": params},
                                                          inputs, targets)

    return logits_batch, None

  def loss_fn(
      self,
      label_batch: spec.Tensor,  # Dense (not one-hot) labels.
      logits_batch: spec.Tensor) -> spec.Tensor:

    weights = jnp.where(label_batch > 0, 1.0, 0.0)
    loss, _ = self.compute_weighted_cross_entropy(logits_batch, label_batch,
                                                  weights)

    return loss

  def output_activation_fn(self, logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:
    """Return the final activations of the model."""
    pass

  def eval_model(self, params: spec.ParameterContainer,
                 model_state: spec.ModelAuxiliaryState, rng: spec.RandomState,
                 data_dir: str):
    """Run a full evaluation of the model."""
    del data_dir

    eval_results = self.evaluate(
        p_eval_step=self._p_eval_step,
        target=params,
        eval_ds=self._eval_ds,
        num_eval_steps=20)

    _, bleu_score = self.translate_and_calculate_bleu(
        p_pred_step=self._p_pred_step,
        p_init_cache=self._p_init_cache,
        target=params,
        predict_ds=self._predict_ds,
        decode_tokens=self._decode_tokens,
        max_predict_length=256)

    eval_results["bleu"] = bleu_score

    return eval_results

