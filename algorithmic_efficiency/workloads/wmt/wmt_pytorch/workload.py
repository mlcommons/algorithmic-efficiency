"""WMT workload implemented in PyTorch."""
import contextlib
from typing import Tuple

from absl import logging
import jax.dlpack
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.wmt import bleu
from algorithmic_efficiency.workloads.wmt import decode
from algorithmic_efficiency.workloads.wmt.wmt_pytorch.models import Transformer
from algorithmic_efficiency.workloads.wmt.workload import BaseWmtWorkload

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _jax_to_pytorch(x: spec.Tensor, take_ownership: bool = False):
  return torch.utils.dlpack.from_dlpack(
      jax.dlpack.to_dlpack(x, take_ownership=take_ownership))


def _pytorch_to_jax(x: spec.Tensor):
  x = x.contiguous()  # https://github.com/google/jax/issues/8082
  return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):

  def forward(self, logits, targets, label_smoothing=0.1):
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
        confidence * np.log(confidence) +
        (vocab_size - 1) * low_confidence * np.log(low_confidence + 1e-20))
    one_hot_targets = F.one_hot(targets, num_classes=vocab_size)
    soft_targets = torch.where(one_hot_targets == 1, confidence, low_confidence)
    loss = super().forward(
        input=logits.transpose(-2, -1), target=soft_targets.transpose(-2, -1))
    return loss - normalizing_constant


class WmtWorkload(BaseWmtWorkload):
  """WMT PyTorch workload."""

  def compute_weighted_cross_entropy(self,
                                     logits,
                                     targets,
                                     weights=None,
                                     label_smoothing=0.1):
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
      raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                       (str(logits.shape), str(targets.shape)))

    loss_fn = CrossEntropyLoss(reduction='none')
    if torch.cuda.device_count() > 1:
      loss_fn = torch.nn.DataParallel(loss_fn)

    loss = loss_fn(logits, targets, label_smoothing=label_smoothing)
    if weights is not None:
      loss = loss * weights
    return loss

  # Primary eval / decode step functions.
  # ----------------------------------------------------------------------------
  @torch.no_grad()
  def predict_step(self, inputs, params, eos_id, max_decode_len, beam_size=4):
    """Predict translation with fast decoding beam search on a batch."""
    # This means that decoding will always happen on a single GPU!
    params = params.module if isinstance(params,
                                         torch.nn.DataParallel) else params
    params.eval()
    encoder = params.encoder
    if torch.cuda.device_count() > 1:
      encoder = torch.nn.DataParallel(encoder)
    encoded_inputs = torch.repeat_interleave(
        encoder(inputs), repeats=beam_size, dim=0)
    raw_inputs = torch.repeat_interleave(inputs, repeats=beam_size, dim=0)
    decoder = params.decoder
    if torch.cuda.device_count() > 1:
      decoder = torch.nn.DataParallel(decoder)

    def tokens_ids_to_logits(flat_ids, flat_cache):
      """Token slice to logits from decoder model."""
      # --> [batch * beam, 1, vocab]
      flat_ids = _jax_to_pytorch(flat_ids)
      flat_logits, new_flat_cache = decoder(
          flat_ids,
          encoded_inputs,
          raw_inputs,
          decode=True,
          max_len=max_decode_len,
          cache=flat_cache)
      # Remove singleton sequence-length dimension:
      # [batch * beam, 1, vocab] --> [batch * beam, vocab]
      flat_logits = _pytorch_to_jax(flat_logits).squeeze(axis=1)
      return flat_logits, new_flat_cache

    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    beam_seqs, _ = decode.beam_search(
        inputs,
        None,
        tokens_ids_to_logits,
        beam_size=beam_size,
        alpha=0.6,
        eos_id=eos_id,
        max_decode_len=max_decode_len,
        lax_while=False)

    # Beam search returns [n_batch, n_beam, n_length + 1] with beam dimension
    # sorted in increasing order of log-probability.
    # Return the highest scoring beam sequence, drop first dummy 0 token.
    return beam_seqs[:, -1, 1:]

  # Utils for prediction and BLEU calculation
  # ----------------------------------------------------------------------------

  def pad_examples(self, x, desired_batch_size):
    """Expand batch to desired size by repeating last slice."""
    batch_pad = desired_batch_size - x.shape[0]
    return torch.cat([x, torch.tile(x[-1], (batch_pad, 1))], dim=0)

  def translate_and_calculate_bleu(self,
                                   params: spec.ParameterContainer,
                                   predict_ds: tf.data.Dataset,
                                   max_predict_length: int,
                                   num_eval_steps: int):
    """Translates the `predict_ds` and calculates the BLEU score."""
    n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logging.info('Translating evaluation dataset.')
    sources, references, predictions = [], [], []
    for _, pred_batch in zip(range(num_eval_steps + 1), predict_ds):
      inputs, targets = self.preprocess_for_eval(pred_batch, None, None)  # pylint: disable=unbalanced-tuple-unpacking
      # Handle final odd-sized batch by padding instead of dropping it.
      cur_pred_batch_size = inputs.shape[0]
      if cur_pred_batch_size % n_devices:
        padded_size = int(np.ceil(cur_pred_batch_size / n_devices) * n_devices)
        inputs = self.pad_examples(inputs, padded_size)  # pylint: disable=cell-var-from-loop
        targets = self.pad_examples(targets, padded_size)
      predicted = self.predict_step(inputs,
                                    params,
                                    decode.EOS_ID,
                                    max_predict_length)

      # Iterate through non-padding examples of batch.
      for i, s in enumerate(predicted[:cur_pred_batch_size]):
        sources.append(self._decode_tokens(inputs[i]))
        references.append(self._decode_tokens(targets[i]))
        predictions.append(self._decode_tokens(s))
    logging.info("Translation: %d predictions %d references %d sources.",
                 len(predictions),
                 len(references),
                 len(sources))

    # Calculate BLEU score for translated eval corpus against reference.
    bleu_matches = bleu.bleu_partial(references, predictions)
    bleu_score = bleu.complete_bleu(*bleu_matches)
    return bleu_score

  def preprocess_for_train(self,
                           selected_raw_input_batch: spec.Tensor,
                           selected_label_batch: spec.Tensor,
                           train_mean: spec.Tensor,
                           train_stddev: spec.Tensor,
                           rng: spec.RandomState,
                           packed_examples: bool = False) -> spec.Tensor:
    del selected_label_batch
    del train_mean
    del train_stddev
    del rng
    inputs = torch.tensor(
        selected_raw_input_batch['inputs'].numpy(),
        device=DEVICE,
        dtype=torch.int)
    targets = torch.tensor(
        selected_raw_input_batch['targets'].numpy(),
        device=DEVICE,
        dtype=torch.int64)
    if packed_examples:
      extras = [
          'inputs_position',
          'targets_position',
          'inputs_segmentation',
          'targets_segmentation'
      ]
      extra_inputs = []
      for extra in extras:
        extra_inputs.append(
            torch.tensor(
                selected_raw_input_batch[extra].numpy(),
                device=DEVICE,
                dtype=torch.int64))
      in_pos, tgt_pos, in_seg, tgt_seg = extra_inputs  # pylint: disable=unbalanced-tuple-unpacking
      return inputs, targets, in_pos, tgt_pos, in_seg, tgt_seg
    return inputs, targets

  def preprocess_for_eval(self,
                          raw_input_batch: spec.Tensor,
                          train_mean: spec.Tensor,
                          train_stddev: spec.Tensor) -> spec.Tensor:
    del train_mean
    del train_stddev
    return self.preprocess_for_train(raw_input_batch, None, None, None, None)

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model = Transformer()
    self._param_shapes = {
        k: spec.ShapeTuple(v.shape) for k, v in model.named_parameters()
    }
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    model.to(DEVICE)
    return model, None

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

    model = params

    if mode == spec.ForwardPassMode.EVAL:
      model.eval()

    contexts = {
        spec.ForwardPassMode.EVAL: torch.no_grad,
        spec.ForwardPassMode.TRAIN: contextlib.nullcontext
    }

    with contexts[mode]():
      logits_batch = model(*augmented_and_preprocessed_input_batch)

    return logits_batch, None

  def loss_fn(
      self,
      label_batch: spec.Tensor,  # Dense (not one-hot) labels.
      logits_batch: spec.Tensor) -> spec.Tensor:
    loss = self.compute_weighted_cross_entropy(logits_batch, label_batch)
    return loss

  def eval_step(self, params, batch):
    """Calculate evaluation metrics on a batch."""
    targets = batch[1]
    weights = torch.where(targets > 0, 1.0, 0.0)
    logits, _ = self.model_fn(
        params,
        batch,
        mode=spec.ForwardPassMode.EVAL,
        model_state=None,
        rng=None,
        update_batch_norm=False)
    return self.compute_metrics(logits, targets, weights)

  def evaluate(self,
               params: spec.ParameterContainer,
               eval_ds: tf.data.Dataset,
               num_eval_steps: int):
    """Evaluate the model and return a dictionary with the metrics."""
    logging.info('Gathering evaluation metrics.')
    eval_metrics = {
        'loss': 0.,
        'accuracy': 0.,
        'denominator': 0,
    }
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    for _, eval_batch in zip(range(num_eval_steps), eval_iter):
      eval_batch = self.preprocess_for_eval(eval_batch, None, None)
      metrics = self.eval_step(params, eval_batch)
      eval_metrics = {k: v + metrics[k] for k, v in eval_metrics.items()}
    denominator = eval_metrics.pop('denominator')
    return {k: float(v / denominator) for k, v in eval_metrics.items()}
