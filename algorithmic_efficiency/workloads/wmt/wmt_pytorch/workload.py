"""WMT workload implemented in PyTorch."""
import contextlib
import os
from typing import Dict, Optional, Tuple

from absl import logging
import jax.dlpack
import numpy as np
import tensorflow as tf
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from algorithmic_efficiency import data_utils
from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.wmt import bleu
from algorithmic_efficiency.workloads.wmt import decode
from algorithmic_efficiency.workloads.wmt.wmt_pytorch.models import Transformer
from algorithmic_efficiency.workloads.wmt.workload import BaseWmtWorkload

USE_PYTORCH_DDP = 'LOCAL_RANK' in os.environ
RANK = int(os.environ['LOCAL_RANK']) if USE_PYTORCH_DDP else 0
DEVICE = torch.device(f'cuda:{RANK}' if torch.cuda.is_available() else 'cpu')
N_GPUS = torch.cuda.device_count()


def _jax_to_pytorch(x: spec.Tensor, take_ownership: bool = False):
  return torch.utils.dlpack.from_dlpack(
      jax.dlpack.to_dlpack(x, take_ownership=take_ownership))


def _pytorch_to_jax(x: spec.Tensor):
  x = x.contiguous()  # https://github.com/google/jax/issues/8082
  return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):

  def forward(self, input, target, label_smoothing=0.1):  # pylint: disable=redefined-builtin
    vocab_size = input.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
        confidence * np.log(confidence) +
        (vocab_size - 1) * low_confidence * np.log(low_confidence + 1e-20))
    one_hot_targets = F.one_hot(target, num_classes=vocab_size)
    soft_targets = torch.where(one_hot_targets == 1, confidence, low_confidence)
    loss = super().forward(
        input=input.transpose(-2, -1), target=soft_targets.transpose(-2, -1))
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
    if N_GPUS > 1 and not USE_PYTORCH_DDP:
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
    params = params.module if isinstance(params, (torch.nn.DataParallel,
                                                  DDP)) else params
    params.eval()
    encoder = params.encoder
    if N_GPUS > 1 and not USE_PYTORCH_DDP:
      encoder = torch.nn.DataParallel(encoder)
    encoded_inputs = torch.repeat_interleave(
        encoder(inputs), repeats=beam_size, dim=0)
    raw_inputs = torch.repeat_interleave(inputs, repeats=beam_size, dim=0)
    decoder = params.decoder
    if N_GPUS > 1 and not USE_PYTORCH_DDP:
      decoder = torch.nn.DataParallel(decoder)

    def tokens_ids_to_logits(flat_ids, flat_cache):
      """Token slice to logits from decoder model."""
      # --> [batch * beam, 1, vocab]
      flat_ids = _jax_to_pytorch(flat_ids).to(DEVICE)
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
                                   ds_iter: tf.data.Dataset,
                                   num_batches: int,
                                   max_predict_length: int):
    """Translates the `ds_iter` and calculates the BLEU score."""
    logging.info('Translating evaluation dataset.')
    sources, references, predictions = [], [], []
    for _ in range(num_batches):
      pred_batch = next(ds_iter)
      inputs = pred_batch['inputs']
      targets = pred_batch['targets']
      cur_pred_batch_size = inputs.shape[0]
      if not USE_PYTORCH_DDP:
        # Handle final odd-sized batch by padding instead of dropping it.
        if cur_pred_batch_size % N_GPUS:
          padded_size = int(np.ceil(cur_pred_batch_size / N_GPUS) * N_GPUS)
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

    # Calculate BLEU score for translated eval corpus against reference.
    bleu_matches = bleu.bleu_partial(references, predictions)
    if USE_PYTORCH_DDP:
      # Sync matches across devices.
      for idx, array in enumerate(bleu_matches):
        tensor = torch.as_tensor(array, device=DEVICE)
        dist.all_reduce(tensor)
        bleu_matches[idx] = tensor.cpu().numpy()
    bleu_score = bleu.complete_bleu(*bleu_matches)
    return bleu_score

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model = Transformer()
    self._param_shapes = {
        k: spec.ShapeTuple(v.shape) for k, v in model.named_parameters()
    }
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = torch.nn.DataParallel(model)
    return model, None

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
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
      logits_batch = model(
          src=augmented_and_preprocessed_input_batch['inputs'],
          tgt=augmented_and_preprocessed_input_batch['targets'],
          inputs_positions=augmented_and_preprocessed_input_batch.get(
              'inputs_position', None),
          targets_positions=augmented_and_preprocessed_input_batch.get(
              'targets_position', None),
          inputs_segmentation=augmented_and_preprocessed_input_batch.get(
              'inputs_segmentation', None),
          targets_segmentation=augmented_and_preprocessed_input_batch.get(
              'targets_segmentation', None))

    return logits_batch, None

  def build_input_queue(self,
                        data_rng: jax.random.PRNGKey,
                        split: str,
                        data_dir: str,
                        global_batch_size: int,
                        num_batches: Optional[int] = None,
                        repeat_final_dataset: bool = False):
    np_iter = super().build_input_queue(data_rng,
                                        split,
                                        data_dir,
                                        global_batch_size,
                                        num_batches,
                                        repeat_final_dataset)
    if USE_PYTORCH_DDP:
      return data_utils.TFDistributedSampler(np_iter, device=DEVICE, rank=RANK)

    def _input_queue_generator():
      for batch in np_iter:
        batch = {
            key: torch.as_tensor(value, device=DEVICE,
                                 dtype=torch.int64).view(-1, value.shape[-1])
            for key,
            value in batch.items()
        }
        yield batch

    return _input_queue_generator()

  def eval_step(self, params, batch):
    """Calculate evaluation metrics on a batch."""
    targets = batch['targets']
    weights = torch.where(targets > 0, 1.0, 0.0)
    logits, _ = self.model_fn(
        params,
        batch,
        mode=spec.ForwardPassMode.EVAL,
        model_state=None,
        rng=None,
        update_batch_norm=False)
    return self.compute_summed_metrics(logits, targets, weights)

  @property
  def model_params_types(self):
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    if self._param_types is None:
      self._param_types = param_utils.jax_param_types(self._param_shapes)
    return self._param_types
