"""WMT workload implemented in PyTorch."""
import contextlib
from typing import Dict, Optional, Tuple

from absl import logging
import jax
import tensorflow as tf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import pytorch_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.interop_utils import jax_to_pytorch
from algorithmic_efficiency.workloads.wmt import bleu
from algorithmic_efficiency.workloads.wmt import decode
from algorithmic_efficiency.workloads.wmt.wmt_pytorch.models import Transformer
from algorithmic_efficiency.workloads.wmt.workload import BaseWmtWorkload

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()


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

    loss_fn = torch.nn.CrossEntropyLoss(
        reduction='none', label_smoothing=label_smoothing)
    if N_GPUS > 1 and not USE_PYTORCH_DDP:
      loss_fn = torch.nn.DataParallel(loss_fn)

    # PyTorch loss functions expect the class dim directly after the batch dim.
    loss = loss_fn(logits.transpose(-2, -1), targets)
    if weights is not None:
      loss = loss * weights
    return loss

  # Primary eval / decode step functions.
  # ----------------------------------------------------------------------------
  @torch.no_grad()
  def predict_step(self, inputs, params, eos_id, max_decode_len, beam_size=4):
    """Predict translation with fast decoding beam search on a batch."""
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
      flat_ids = jax_to_pytorch(flat_ids).to(DEVICE)
      flat_logits, new_flat_cache = decoder(
          flat_ids,
          encoded_inputs,
          raw_inputs,
          decode=True,
          max_len=max_decode_len,
          cache=flat_cache)
      # Remove singleton sequence-length dimension:
      # [batch * beam, 1, vocab] --> [batch * beam, vocab]
      flat_logits = flat_logits.cpu().numpy().squeeze(axis=1)
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

  def translate_and_calculate_bleu(self,
                                   params: spec.ParameterContainer,
                                   ds_iter: tf.data.Dataset,
                                   num_batches: int,
                                   max_predict_length: int):
    """Translates the `ds_iter` and calculates the BLEU score."""
    logging.info('Translating evaluation dataset.')
    references, predictions = [], []
    for _ in range(num_batches):
      pred_batch = next(ds_iter)
      inputs = pred_batch['inputs']
      targets = pred_batch['targets']
      predicted = self.predict_step(inputs,
                                    params,
                                    decode.EOS_ID,
                                    max_predict_length)

      # Iterate through non-padding examples of batch.
      assert len(predicted) == len(targets)
      for tar, pred in zip(targets, predicted):
        references.append(self._decode_tokens(tar))
        predictions.append(self._decode_tokens(pred))

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

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """aux_dropout_rate is used as attention_dropout_rate."""
    torch.random.manual_seed(rng[0])
    model = Transformer(
        dropout_rate=dropout_rate, attention_dropout_rate=aux_dropout_rate)
    self._param_shapes = param_utils.pytorch_param_shapes(model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = torch.nn.DataParallel(model)
    return model, None

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key == 'shared_embedding.weight'

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

  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         num_batches: Optional[int] = None,
                         repeat_final_dataset: bool = False):
    per_device_batch_size = int(global_batch_size / N_GPUS)
    n_inputs = 6 if split == 'train' else 2

    # The input pipeline has to be created in all processes, because
    # self._tokenizer has to be available in every process.
    np_iter = super()._build_input_queue(data_rng,
                                         split,
                                         data_dir,
                                         global_batch_size,
                                         num_batches,
                                         repeat_final_dataset)
    while True:
      # Only iterate over tf input pipeline in one Python process to
      # avoid creating too many threads.
      if RANK == 0:
        batch = next(np_iter)  # pylint: disable=stop-iteration-return
        tensor_list = []
        for key, value in batch.items():
          tensor = torch.as_tensor(value, dtype=torch.int64, device=DEVICE)
          tensor_list.append(tensor)
          batch[key] = (
              tensor[0] if USE_PYTORCH_DDP else tensor.view(
                  -1, value.shape[-1]))
        # Send batch to other devices when using DDP.
        if USE_PYTORCH_DDP:
          # During eval, the batch size of the remainder might be different.
          if split != 'train':
            per_device_batch_size = torch.tensor(
                len(batch['inputs']), dtype=torch.int32, device=DEVICE)
            dist.broadcast(per_device_batch_size, src=0)
          dist.broadcast(torch.stack(tensor_list), src=0)
      else:
        # During eval, the batch size of the remainder might be different.
        if split != 'train':
          per_device_batch_size = torch.empty((1,),
                                              dtype=torch.int32,
                                              device=DEVICE)
          dist.broadcast(per_device_batch_size, src=0)
        tensor = torch.empty((n_inputs, N_GPUS, per_device_batch_size, 256),
                             dtype=torch.int64,
                             device=DEVICE)
        dist.broadcast(tensor, src=0)
        # Note that the order of the keys is important.
        if split == 'train':
          keys = [
              'inputs',
              'inputs_position',
              'inputs_segmentation',
              'targets',
              'targets_position',
              'targets_segmentation'
          ]
        # For all eval/test splits.
        else:
          keys = ['inputs', 'targets']
        batch = {}
        for key, n in zip(keys, range(n_inputs)):
          batch[key] = tensor[n][RANK]
      yield batch

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
