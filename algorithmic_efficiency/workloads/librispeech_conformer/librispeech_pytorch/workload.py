"""Conformer workload implemented in PyTorch."""

import contextlib
import functools
import math
import random
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from algorithmic_efficiency import data_utils
from algorithmic_efficiency import param_utils
from algorithmic_efficiency import pytorch_utils
from algorithmic_efficiency import spec
import algorithmic_efficiency.random_utils as prng
from algorithmic_efficiency.workloads.librispeech_conformer import metrics
from algorithmic_efficiency.workloads.librispeech_conformer import workload
from algorithmic_efficiency.workloads.librispeech_conformer.input_pipeline import \
    LibriSpeechDataset
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch import \
    models as conformer_model

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()

MAX_INPUT_LENGTH = 320000


class LibriSpeechConformerWorkload(workload.BaseLibrispeechWorkload):

  def __init__(self,
               tokenizer_vocab_path: Optional[str] = None,
               use_specaug: bool = True) -> None:
    super().__init__()
    self.tokenizer = metrics.load_tokenizer(tokenizer_vocab_path)
    self.use_specaug = use_specaug

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """Conformer model init function.

    Here we use dropout_rate as residual_dropout_rate, and aux_dropout_rate as
    input_dropout_rate.
    """
    torch.random.manual_seed(rng[0])
    # Configure torch backends to avoid OOM errors.
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    model = conformer_model.ConformerEncoderDecoder(
        conformer_model.ConformerConfig(
            attention_residual_dropout_rate=dropout_rate,
            feed_forward_residual_dropout_rate=dropout_rate,
            conv_residual_dropout_rate=dropout_rate,
            input_dropout_rate=aux_dropout_rate,
            use_specaug=self.use_specaug))
    self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none')
    conformer_model.initialize(model)
    self._param_shapes = param_utils.pytorch_param_shapes(model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    model.to(DEVICE)
    self.requires_sync_before_eval = False
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        self.requires_sync_before_eval = True
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = torch.nn.DataParallel(model)
    return model, None

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key in ['lin.weight', 'lin.bias']

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

    model = params
    if mode == spec.ForwardPassMode.EVAL:
      model.eval()
    if mode == spec.ForwardPassMode.TRAIN:
      model.train()
      model.apply(
          functools.partial(
              pytorch_utils.update_batch_norm_fn,
              update_batch_norm=update_batch_norm))

    contexts = {
        spec.ForwardPassMode.EVAL: torch.no_grad,
        spec.ForwardPassMode.TRAIN: contextlib.nullcontext,
    }
    with contexts[mode]():
      inputs, input_paddings = augmented_and_preprocessed_input_batch['inputs']
      logits, logits_paddings = model(inputs.to(DEVICE),
                                      input_paddings.to(DEVICE))
    return (logits, logits_paddings), None

  def _build_input_queue(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      global_batch_size: int,
      cache: Optional[bool] = None,
      repeat_final_dataset: Optional[bool] = None,
      num_batches: Optional[int] = None) -> Iterator[Dict[str, spec.Tensor]]:
    del cache
    del repeat_final_dataset
    del num_batches

    is_train = split == 'train'
    if split == 'train':
      ds_split = 'train-clean-100+train-clean-360+train-other-500'
    elif split == 'eval_train':
      ds_split = 'train-clean-100+train-clean-360+train-other-500'
    elif split == 'validation':
      ds_split = 'dev-clean+dev-other'
    else:
      ds_split = 'test-clean'

    ds = LibriSpeechDataset(split=ds_split, data_dir=data_dir)
    if split == 'eval_train':
      indices = list(range(len(ds)))
      random.Random(data_rng[0]).shuffle(indices)
      ds = torch.utils.data.Subset(ds, indices[:self.num_eval_train_examples])

    sampler = None
    if USE_PYTORCH_DDP:
      per_device_batch_size = global_batch_size // N_GPUS
      ds_iter_batch_size = per_device_batch_size
    else:
      ds_iter_batch_size = global_batch_size
    if USE_PYTORCH_DDP:
      if is_train:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds, num_replicas=N_GPUS, rank=RANK, shuffle=True)
      else:
        sampler = data_utils.DistributedEvalSampler(
            ds, num_replicas=N_GPUS, rank=RANK, shuffle=False)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=ds_iter_batch_size,
        shuffle=not USE_PYTORCH_DDP and is_train,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=is_train)

    dataloader = data_utils.cycle(
        dataloader, custom_sampler=USE_PYTORCH_DDP, use_mixup=False)
    return dataloader

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
    targets, target_paddings = label_batch
    logits, logit_paddings = logits_batch
    logprobs = torch.log_softmax(logits, dim=-1)
    input_lengths = torch.einsum('bh->b', 1 - logit_paddings).long()
    target_lengths = torch.einsum('bh->b', 1 - target_paddings).long()
    per_example_losses = self.ctc_loss(
        logprobs.permute(1, 0, 2),
        targets.long(),
        input_lengths,
        target_lengths)
    # mask_batch is assumed to be shape [batch].
    if mask_batch is not None:
      per_example_losses *= mask_batch
      mask_batch = torch.logical_and(mask_batch, target_lengths)
    else:
      mask_batch = target_lengths
    n_valid_examples = mask_batch.sum().to(per_example_losses)
    summed_loss = per_example_losses.sum()
    n_valid_examples = max(n_valid_examples, 1)
    return {
        'summed': summed_loss,
        'n_valid_examples': torch.as_tensor(n_valid_examples, device=DEVICE),
        'per_example': per_example_losses,
    }

  def greedy_decode(
      self, logits: spec.Tensor,
      logit_paddings: spec.Tensor) -> Tuple[spec.Tensor, spec.Tensor]:
    framewise_tokens = logits.max(dim=-1)[1]
    framewise_tokens = framewise_tokens * (1 - logit_paddings)

    # Add sentinel because unique_consecutive will flatten array
    # and then compute the unique.
    framewise_tokens = torch.cat(
        [framewise_tokens, -torch.ones_like(framewise_tokens[:, 0:1])], dim=1)
    _, indices = torch.unique_consecutive(framewise_tokens, return_inverse=True)
    indices -= indices.min(dim=1, keepdims=True)[0]
    result = torch.zeros_like(framewise_tokens)
    result = result.scatter_(1, indices, framewise_tokens)

    # Replace the sentinel column with 0s and remove it.
    result[result == -1] = 0
    result = result[:, :-1]

    # Remove blanks (id = 0).
    blank_id = 0
    fin_result = torch.zeros_like(result)
    idxs = torch.arange(
        fin_result.numel(), device=result.device).view(*fin_result.shape)
    mask = torch.arange(
        fin_result.shape[1], device=result.device).view(
            1, -1) < result.count_nonzero(dim=1).view(-1, 1)
    fin_result.view(-1)[idxs[mask != 0]] = result[result != blank_id]
    padding = fin_result == 0
    return fin_result, padding

  def sync_sd(self, params: spec.ParameterContainer) -> None:
    sd = params.state_dict()
    dist.barrier()
    for k in sd:
      dist.all_reduce(sd[k], op=dist.ReduceOp.SUM)
      # Assumes N_GPUS is the world size.
      sd[k] = sd[k] / N_GPUS
    params.load_state_dict(sd)

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
    data_rng, model_rng = prng.split(rng, 2)
    if split not in self._eval_iters:
      # These iterators repeat indefinitely.
      self._eval_iters[split] = (
          self._build_input_queue(
              data_rng, split, data_dir, global_batch_size=global_batch_size))

    total_metrics = {
        'loss': torch.tensor(0., device=DEVICE),
        'lengths': torch.tensor(0., device=DEVICE),
        'word_errors': torch.tensor(0., device=DEVICE),
        'num_words': torch.tensor(0., device=DEVICE),
    }
    num_batches = int(math.ceil(num_examples / global_batch_size))
    if self.requires_sync_before_eval:
      self.sync_sd(params)
    for _ in range(num_batches):
      batch = next(self._eval_iters[split])

      (logits, logits_padding), _ = self.model_fn(
          params,
          batch,
          model_state,
          spec.ForwardPassMode.EVAL,
          model_rng,
          update_batch_norm=False)
      decoded, decoded_paddings = self.greedy_decode(logits, logits_padding)
      targets, target_paddings = batch['targets']
      word_errors, num_words = metrics.compute_wer(
          decoded=decoded.cpu().numpy(),
          decoded_paddings=decoded_paddings.cpu().numpy(),
          targets=targets.cpu().numpy(),
          target_paddings=target_paddings.cpu().numpy(),
          tokenizer=self.tokenizer)
      loss = self.loss_fn((targets, target_paddings), (logits, logits_padding))
      summed_loss = loss['summed']
      lengths = loss['n_valid_examples']
      batch_metrics = {
          'loss': summed_loss,
          'lengths': lengths,
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
