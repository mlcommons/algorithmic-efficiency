import contextlib
import itertools
import math
from typing import Dict, Optional, Tuple

from absl import logging
import jax
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import pytorch_utils
from algorithmic_efficiency import spec
import algorithmic_efficiency.random_utils as prng
from algorithmic_efficiency.workloads.librispeech_conformer import metrics
from algorithmic_efficiency.workloads.librispeech_conformer import workload
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch import \
    model as conformer_model

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()

MAX_INPUT_LENGTH = 320000


def _maybe_update_model_dropout(model,
                                residual_dropout_rate,
                                input_dropout_rate):
  for child in list(model.modules()):
    # Residual dropout.
    if (isinstance(child, conformer_model.MultiHeadedSelfAttention) and
        residual_dropout_rate is not None):
      child.dropout.p = residual_dropout_rate
    elif (isinstance(child, conformer_model.ConvolutionBlock) and
          residual_dropout_rate is not None):
      child.dropout.p = residual_dropout_rate
    elif (isinstance(child, conformer_model.FeedForwardModule) and
          residual_dropout_rate is not None):
      child.dropout2.p = residual_dropout_rate
    # Input dropout.
    elif (isinstance(child, conformer_model.Subsample) and
          input_dropout_rate is not None):
      child.dropout.p = input_dropout_rate


class LibriSpeechConformerWorkload(workload.BaseLibrispeechWorkload):

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model = conformer_model.ConformerEncoderDecoder(
        conformer_model.ConformerConfig())
    self._model = model
    self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none')
    # Run model once to initialize lazy layers
    t = MAX_INPUT_LENGTH
    wave = torch.randn((2, t))
    pad = torch.zeros_like(wave)
    _ = model(wave, pad)
    conformer_model.initialize(model)
    self._param_shapes = param_utils.pytorch_param_shapes(model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = torch.nn.DataParallel(model)
    return model, None

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def init_tokenizer(self, tokenizer_vocab_path):
    logging.info('Initializing tokenizer.')
    self.tokenizer = metrics.load_tokenizer(tokenizer_vocab_path)

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      dropout_rate: Optional[float],
      aux_dropout_rate: Optional[float],
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    """Conformer model function.

    Here we use dropout_rate as residual_dropout_rate, and aux_dropout_rate as
    input_dropout_rate.
    """
    del model_state
    del rng
    del update_batch_norm

    model = params
    _maybe_update_model_dropout(
        model,
        residual_dropout_rate=dropout_rate,
        input_dropout_rate=aux_dropout_rate)

    if mode == spec.ForwardPassMode.EVAL:
      model.eval()

    if mode == spec.ForwardPassMode.TRAIN:
      model.train()

    contexts = {
        spec.ForwardPassMode.EVAL: torch.no_grad,
        spec.ForwardPassMode.TRAIN: contextlib.nullcontext
    }

    with contexts[mode]():
      inputs, input_paddings = augmented_and_preprocessed_input_batch['inputs']
      logits, logits_paddings = model(inputs, input_paddings)

    return (logits, logits_paddings), None

  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         num_batches: Optional[int] = None,
                         repeat_final_dataset: bool = False):
    per_device_batch_size = int(global_batch_size / N_GPUS)
    keys = ['inputs', 'targets']
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
        for key in keys:
          value, value_paddings = batch[key]
          tensor = torch.as_tensor(value, dtype=torch.float32, device=DEVICE)
          tensor_paddings = torch.as_tensor(
              value_paddings, dtype=torch.float32, device=DEVICE)
          tensor_list.append(tensor)
          tensor_list.append(tensor_paddings)
          if USE_PYTORCH_DDP:
            batch[key] = (tensor[0], tensor_paddings[0])
          else:
            batch[key] = (tensor.view(-1, *value.shape[2:]),
                          tensor_paddings.view(-1, *value_paddings.shape[2:]))
        # Send batch to other devices when using DDP.
        if USE_PYTORCH_DDP:
          # During eval, the batch size of the remainder might be different.
          if split != 'train':
            per_device_batch_size = torch.tensor(
                len(batch['inputs'][0]), dtype=torch.int32, device=DEVICE)
            dist.broadcast(per_device_batch_size, src=0)
          dist.broadcast(torch.cat(tensor_list, dim=-1), src=0)
      else:
        # During eval, the batch size of the remainder might be different.
        if split != 'train':
          per_device_batch_size = torch.empty((1,),
                                              dtype=torch.int32,
                                              device=DEVICE)
          dist.broadcast(per_device_batch_size, src=0)
        tensor = torch.empty(
            (N_GPUS, per_device_batch_size, MAX_INPUT_LENGTH * 2 + 256 * 2),
            dtype=torch.float32,
            device=DEVICE)
        dist.broadcast(tensor, src=0)
        # Note that the order of the keys is important.
        tensors = tensor.split([MAX_INPUT_LENGTH, MAX_INPUT_LENGTH, 256, 256],
                               dim=-1)
        batch = {
            'inputs': (tensors[0][RANK], tensors[1][RANK]),
            'targets': (tensors[2][RANK], tensors[3][RANK]),
        }
      yield batch

  def _loss_fn(
      self,
      label_batch: Tuple[spec.Tensor, spec.Tensor],
      logits_batch: Tuple[spec.Tensor, spec.Tensor]
  ) -> spec.Tensor:  # differentiable
    targets, target_paddings = label_batch
    logits, logit_paddings = logits_batch
    logprobs = torch.log_softmax(logits, dim=-1)
    input_lengths = torch.einsum('bh->b', 1 - logit_paddings).long()
    target_lengths = torch.einsum('bh->b', 1 - target_paddings).long()
    per_seq_loss = self.ctc_loss(
        logprobs.permute(1, 0, 2),
        targets.long(),
        input_lengths,
        target_lengths)
    average_loss = per_seq_loss.sum() / target_lengths.sum()
    return {
        'loss': per_seq_loss.sum(),
        'lengths': target_lengths.sum(),
        'average_loss': average_loss
    }

  def loss_fn(self,
              label_batch: Tuple[spec.Tensor, spec.Tensor],
              logits_batch: Tuple[spec.Tensor, spec.Tensor],
              mask_batch: Optional[spec.Tensor] = None,
              label_smoothing: float = 0.0) -> spec.Tensor:  # differentiable
    del mask_batch
    del label_smoothing
    return self._loss_fn(label_batch, logits_batch)['average_loss']

  def greedy_decode(self, logits, logit_paddings):
    framewise_tokens = logits.max(dim=-1)[1]
    framewise_tokens = framewise_tokens * (logit_paddings)

    # add sentinel because unique_consecutive will flatten array
    # and then compute the unique
    framewise_tokens = torch.cat(
        [framewise_tokens, -torch.ones_like(framewise_tokens[:, 0:1])], dim=1)
    _, indices = torch.unique_consecutive(framewise_tokens, return_inverse=True)
    indices -= indices.min(dim=1, keepdims=True)[0]
    result = torch.zeros_like(framewise_tokens)
    result = result.scatter_(1, indices, framewise_tokens)

    # replace the sentinel column with 0s and remove it
    result[result == -1] = 0
    result = result[:, :-1]

    # remove blanks (id = 0)
    blank_id = 0
    fin_result = torch.zeros_like(result)
    idxs = torch.arange(
        fin_result.numel(), device=result.device).view(*fin_result.shape)
    mask = torch.arange(
        fin_result.shape[1], device=result.device).view(
            1, -1) < result.count_nonzero(dim=1).view(-1, 1)
    fin_result.view(-1)[idxs[mask != 0]] = result[result != blank_id]
    padding = (fin_result == 0)
    return fin_result, padding

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str,
                           global_step: int):
    """Run a full evaluation of the model."""
    del global_step
    data_rng, model_rng = prng.split(rng, 2)
    if split not in self._eval_iters:
      # These iterators repeat indefinitely.
      self._eval_iters[split] = itertools.cycle(
          self._build_input_queue(
              data_rng, split, data_dir, global_batch_size=global_batch_size))

    total_metrics = {
        'loss': torch.tensor(0., device=DEVICE),
        'lengths': torch.tensor(0., device=DEVICE),
        'word_errors': torch.tensor(0., device=DEVICE),
        'num_words': torch.tensor(0., device=DEVICE),
    }
    num_batches = int(math.ceil(num_examples / global_batch_size))

    for _ in range(num_batches):
      batch = next(self._eval_iters[split])

      (logits, logits_padding), _ = self.model_fn(
          params,
          batch,
          model_state,
          spec.ForwardPassMode.EVAL,
          model_rng,
          dropout_rate=0.1,  # Default, unused for eval.
          aux_dropout_rate=0.1,  # Default, unused for eval.
          update_batch_norm=False)
      decoded, decoded_paddings = self.greedy_decode(logits, logits_padding)
      targets, target_paddings = batch['targets']
      word_errors, num_words = metrics.compute_wer(
          decoded=decoded.detach().cpu().numpy(),
          decoded_paddings=decoded_paddings.detach().cpu().numpy(),
          targets=targets.detach().cpu().numpy(),
          target_paddings=target_paddings.detach().cpu().numpy(),
          tokenizer=self.tokenizer)
      loss = self._loss_fn((targets, target_paddings), (logits, logits_padding))
      batch_metrics = {
          'loss': loss['loss'],
          'lengths': loss['lengths'],
          'word_errors': word_errors,
          'num_words': num_words,
      }
      total_metrics = {
          k: v + batch_metrics[k] for k, v in total_metrics.items()
      }
    if USE_PYTORCH_DDP:
      for metric in total_metrics.values():
        dist.all_reduce(metric)
    return {
        'ctc_loss':
            float(total_metrics['loss'].item() /
                  total_metrics['lengths'].item()),
        'wer':
            float(total_metrics['word_errors'].item() /
                  total_metrics['num_words'].item()),
    }
