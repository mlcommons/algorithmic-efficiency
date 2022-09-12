import functools
import itertools
import math
from typing import Dict, Tuple

from absl import flags
from absl import logging
from flax import jax_utils
import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech_conformer import metrics
from algorithmic_efficiency.workloads.librispeech_conformer import workload
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax import \
    models

FLAGS = flags.FLAGS


class LibriSpeechConformerWorkload(workload.BaseLibrispeechWorkload):

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    model_cls = getattr(models, 'Conformer')
    model = model_cls(models.ConformerConfig())
    self._model = model
    input_shape = [(320000,), (320000,)]
    fake_input_batch = [np.zeros((2, *x), jnp.float32)
                        for x in input_shape]

    model_init_fn = jax.jit(functools.partial(model.init, train=False))

    params_rng, dropout_rng = jax.random.split(rng, 2)
    variables = model_init_fn(
        {'params': params_rng, 'dropout': dropout_rng}, *fake_input_batch)

    model_state, params = variables.pop('params')

    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      params)
    model_state = jax_utils.replicate(model_state)
    params = jax_utils.replicate(params)
    return params, model_state

  def init_metrics_bundle(self, tokenizer_vocab_path):
    logging.info('Initializing metrics bundle.')
    self.metrics_bundle = metrics.get_metrics_bundle(tokenizer_vocab_path)

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState = None
  ) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    variables = {'params': params, **model_state}

    is_train_mode = (mode == spec.ForwardPassMode.TRAIN)

    if is_train_mode:
      (logits, logit_paddings), new_model_state = self._model.apply(
          variables,
          augmented_and_preprocessed_input_batch['inputs'],
          augmented_and_preprocessed_input_batch['input_paddings'],
          is_train_mode,
          rngs=rng,
          mutable=['batch_stats'])
      return (logits, logit_paddings), new_model_state
    else:
      logits, logit_paddings = self._model.apply(
          variables,
          augmented_and_preprocessed_input_batch['inputs'],
          augmented_and_preprocessed_input_batch['input_paddings'],
          is_train_mode,
          mutable=False)
      return logits, logit_paddings

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

  def loss_fn(self, logits, logit_paddings, targets,
              target_paddings):  # differentiable
    logprobs = nn.log_softmax(logits)
    per_seq_loss = self.ctc_loss(logprobs,
                                  logit_paddings,
                                  targets,
                                  target_paddings)
    normalizer = jnp.sum(1 - target_paddings)

    normalized_loss = jnp.sum(per_seq_loss) / jnp.maximum(normalizer, 1)
    return normalized_loss

  def ctc_loss(self,
                logits,
                logit_paddings,
                labels,
                label_paddings,
                blank_id=0):
    return optax.ctc_loss(logits,
                          logit_paddings,
                          labels,
                          label_paddings,
                          blank_id)

  # Adapted from lingvo's greedy decoding logic here:
  # https://github.com/tensorflow/lingvo/blob/2ee26814c57b7dcead3f0382170f2f3da006f810/lingvo/jax/layers/ctc_objectives.py#L138
  def sequence_mask(self, lengths, maxlen):
    batch_size = lengths.shape[0]
    a = jnp.ones([batch_size, maxlen])
    b = jnp.cumsum(a, axis=-1)
    c = jnp.less_equal(b, lengths[:, jnp.newaxis]).astype(lengths.dtype)
    return c

  def collapse_and_remove_blanks(
          self,
          labels,
          seq_length,
          blank_id: int = 0):
    b, t = labels.shape
    # Zap out blank
    blank_mask = 1 - jnp.equal(labels, blank_id)
    labels = (labels * blank_mask).astype(labels.dtype)

    # Mask labels that don't equal previous label.
    label_mask = jnp.concatenate([
        jnp.ones_like(labels[:, :1], dtype=jnp.int32),
        jnp.not_equal(labels[:, 1:], labels[:, :-1])
    ], axis=1)

    # Filter labels that aren't in the original sequence.
    maxlen = labels.shape[1]
    seq_mask = self.sequence_mask(seq_length, maxlen=maxlen)
    label_mask = label_mask * seq_mask

    # remove repetitions from the labels
    ulabels = label_mask * labels

    # Count masks for new sequence lengths.
    label_mask = jnp.not_equal(ulabels, 0).astype(labels.dtype)
    new_seq_len = jnp.sum(label_mask, axis=1)

    # Mask indexes based on sequence length mask.
    new_maxlen = maxlen
    idx_mask = self.sequence_mask(new_seq_len, maxlen=new_maxlen)

    # Flatten everything and mask out labels to keep and sparse indices.
    flat_labels = jnp.reshape(ulabels, [-1])
    flat_idx_mask = jnp.reshape(idx_mask, [-1])

    indices = jnp.nonzero(flat_idx_mask, size=b * t)[0]
    values = jnp.nonzero(flat_labels, size=b * t)[0]
    updates = jnp.take_along_axis(flat_labels, values, axis=-1)

    # Scatter to flat shape.
    flat = jnp.zeros(flat_idx_mask.shape).astype(labels.dtype)
    flat = flat.at[indices].set(updates)
    # 0'th position in the flat array gets clobbered by later padded updates,
    # so reset it here to its original value
    flat = flat.at[0].set(updates[0])

    # Reshape back to square batch.
    batch_size = labels.shape[0]
    new_shape = [batch_size, new_maxlen]
    return (jnp.reshape(flat, new_shape).astype(labels.dtype),
            new_seq_len.astype(seq_length.dtype))

  def greedy_decode(self, logits, logit_paddings):
    per_frame_max = jnp.argmax(logits, axis=-1)
    seqlen = jnp.sum(1.0 - logit_paddings, axis=-1)
    hyp, _ = self.collapse_and_remove_blanks(
        per_frame_max, seqlen, blank_id=0)
    hyp_paddings = jnp.equal(hyp, 0).astype(jnp.int32)
    return hyp, hyp_paddings

  @functools.partial(
      jax.pmap,
      axis_name='batch',
      in_axes=(None, 0, 0, 0, None),
      static_broadcasted_argnums=(0,))
  def eval_step_pmapped(
          self,
          params: spec.ParameterContainer,
          batch: Dict[str, spec.Tensor],
          model_state: spec.ModelAuxiliaryState,
          rng: spec.RandomState) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:  # pylint: disable=line-too-long
    logits, logit_paddings = self.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.EVAL)

    decoded, decoded_paddings = self.greedy_decode(logits, logit_paddings)
    normalized_loss = self.loss_fn(logits,
                                    logit_paddings,
                                    batch['targets'],
                                    batch['target_paddings'])

    return self.metrics_bundle.gather_from_model_output(
        normalized_loss=normalized_loss,
        decoded=decoded,
        decoded_paddings=decoded_paddings,
        targets=batch['targets'],
        target_paddings=batch['target_paddings'],
        axis_name='batch')

  def _eval_model_on_split(self,
                            split: str,
                            num_examples: int,
                            global_batch_size: int,
                            params: spec.ParameterContainer,
                            model_state: spec.ModelAuxiliaryState,
                            rng: spec.RandomState,
                            data_dir: str,
                            global_step: int) -> Dict[str, float]:
    """Run a full evaluation of the model."""
    if model_state is not None:
      # Sync batch statistics across replicas before evaluating.
      logging.info('syncing batch_stats across replicas before eval.')
      model_state = self.sync_batch_stats(model_state)

    num_batches = int(math.ceil(num_examples / global_batch_size))
    logging.info('split = %s, num_examples = %d, num_batches = %d',
        split, num_examples, num_batches)

    if split not in self._eval_iters:
      self._eval_iters[split] = itertools.cycle(
        self.build_input_queue(rng,
                              split,
                              data_dir,
                              global_batch_size,
                              num_batches))

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

    self.summary_writer.scalar('{}/WER'.format(split),
                                computed_metrics['wer'],
                                global_step)
    self.summary_writer.scalar('{}/ctc_loss'.format(split),
                                computed_metrics['ctc_loss'],
                                global_step)

    return computed_metrics

  def sync_batch_stats(self, model_state):
      # An axis_name is passed to pmap which can then be used by pmean.
      # In this case each device has its own version of the batch statistics and
      # we average them.
    avg_fn = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')
    new_model_state = model_state.copy(
        {'batch_stats': avg_fn(model_state['batch_stats'])})
    return new_model_state
