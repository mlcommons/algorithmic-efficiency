"""WMT workload implemented in PyTorch."""

import contextlib
from typing import Any, Dict, Optional, Tuple

from absl import logging
import jax
import sacrebleu
import tensorflow as tf
import torch
import torch.distributed as dist
from torch.nn import DataParallel as DP
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import pytorch_utils
from algorithmic_efficiency import spec
#from algorithmic_efficiency.workloads.wmt import bleu
from algorithmic_efficiency.workloads.wmt.wmt_pytorch import decode
from algorithmic_efficiency.workloads.wmt.wmt_pytorch.models import Transformer
from algorithmic_efficiency.workloads.wmt.workload import BaseWmtWorkload

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()


class WmtWorkload(BaseWmtWorkload):
  """WMT PyTorch workload."""

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
     weights: None or array of shape [batch, length].
     label_smoothing: label smoothing constant, used to determine the on and off
       values.

    Returns:
      {'summed': scalar summed loss, 'n_valid_examples': scalar number of
      valid examples in batch, 'per_example': 1-d array of per-example losses}
    """
    if logits.ndim != targets.ndim + 1:
      raise ValueError(f'Incorrect shapes. Got shape {logits.shape} logits and '
                       f'{targets.shape} targets.')

    loss_fn = torch.nn.CrossEntropyLoss(
        reduction='none', label_smoothing=label_smoothing)
    if N_GPUS > 1 and not USE_PYTORCH_DDP:
      loss_fn = DP(loss_fn)

    # PyTorch loss functions expect the class dim directly after the batch dim.
    per_example_losses = loss_fn(logits.transpose(-2, -1), targets)
    if weights is None:
      weights = torch.ones_like(targets)
    per_example_losses = torch.where(
        weights.to(torch.bool), per_example_losses, 0.)
    summed_loss = per_example_losses.sum()
    n_valid_examples = weights.sum()
    return {
        'summed': summed_loss,
        'n_valid_examples': n_valid_examples,
        'per_example': per_example_losses,
    }

  # Primary eval / decode step functions.
  # ----------------------------------------------------------------------------
  @torch.no_grad()
  def predict_step(self,
                   inputs: spec.Tensor,
                   params: spec.ParameterContainer,
                   eos_id: int,
                   max_decode_len: int,
                   beam_size: int = 4) -> spec.Tensor:
    """Predict translation with fast decoding beam search on a batch."""
    # params = params.module if isinstance(params, (DP, DDP)) else params
    if hasattr(params, 'module'):
      params = params.module
    params.eval()

    if hasattr(params, '_modules'):
      params = params._modules
      encoder = params["encoder"]
      decoder = params["decoder"]
    else:
      encoder = params.encoder
      decoder = params.decoder

    if N_GPUS > 1 and not USE_PYTORCH_DDP:
      encoder = DP(encoder)
    if N_GPUS > 1 and not USE_PYTORCH_DDP:
      decoder = DP(decoder)

    encoded_inputs = torch.repeat_interleave(
        encoder(inputs), repeats=beam_size, dim=0)
    raw_inputs = torch.repeat_interleave(inputs, repeats=beam_size, dim=0)

    def tokens_ids_to_logits(
        flat_ids: spec.Tensor, flat_cache: Dict[str, spec.Tensor]
    ) -> Tuple[spec.Tensor, Dict[str, spec.Tensor]]:
      """Token slice to logits from decoder model."""
      # --> [batch * beam, 1, vocab]
      flat_logits, new_flat_cache = decoder(
          flat_ids,
          encoded_inputs,
          raw_inputs,
          decode=True,
          max_len=max_decode_len,
          cache=flat_cache)
      # Remove singleton sequence-length dimension:
      # [batch * beam, 1, vocab] --> [batch * beam, vocab]
      flat_logits = flat_logits.squeeze(dim=1)
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
        max_decode_len=max_decode_len)

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

      # Find actual batch size, ignoring the potential padding.
      weights = pred_batch.get('weights')
      if weights is not None:
        actual_batch_size = weights.sum(0)[0].item()
      else:
        actual_batch_size = len(predicted)
      # Iterate through non-padding examples of batch.
      for idx in range(actual_batch_size):
        references.append(self._decode_tokens(targets[idx]))
        predictions.append(self._decode_tokens(predicted[idx]))

    # Calculate BLEU score for translated eval corpus against reference.
    bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
    return bleu_score

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """aux_dropout_rate is used as attention_dropout_rate."""
    torch.random.manual_seed(rng[0])

    if self.activation == 'relu':
      activation = F.relu
    elif self.activation == 'tanh':
      activation = F.tanh
    else:
      raise ValueError(f'Unknown activation function {self.activation}.')

    model = Transformer(
        dropout_rate=dropout_rate,
        attention_dropout_rate=aux_dropout_rate,
        pre_ln=self.pre_ln,
        attention_temp=self.attention_temp,
        activation=activation,
        glu=self.glu)
    self._param_shapes = param_utils.pytorch_param_shapes(model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = DP(model)
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
        spec.ForwardPassMode.TRAIN: contextlib.nullcontext,
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
    n_inputs = 7 if split == 'train' else 3

    # The input pipeline has to be created in all processes, because
    # self._tokenizer has to be available in every process.
    np_iter = super()._build_input_queue(data_rng,
                                         split,
                                         data_dir,
                                         global_batch_size,
                                         num_batches,
                                         repeat_final_dataset)
    # We only need np_iter in one Python process.
    if RANK != 0:
      del np_iter

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
          # We don't need to broadcast the batch for the device with RANK == 0.
          dist.broadcast(torch.stack(tensor_list)[:, 1:].contiguous(), src=0)
      else:
        batch = {}
        # During eval, the batch size of the remainder might be different.
        if split != 'train':
          per_device_batch_size = torch.empty((1,),
                                              dtype=torch.int32,
                                              device=DEVICE)
          dist.broadcast(per_device_batch_size, src=0)
        # N_GPUS - 1 since we don't broadcast the batch for RANK == 0.
        tensor = torch.empty((n_inputs, N_GPUS - 1, per_device_batch_size, 256),
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
              'targets_segmentation',
              'weights',
          ]
        # For all eval/test splits.
        else:
          keys = ['inputs', 'targets', 'weights']
        for key, n in zip(keys, range(n_inputs)):
          # RANK - 1 since we don't broadcast the batch for RANK == 0.
          batch[key] = tensor[n][RANK - 1]
      yield batch

  def eval_step(self,
                params: spec.ParameterContainer,
                batch: Dict[str, spec.Tensor]) -> Dict[str, spec.Tensor]:
    """Calculate evaluation metrics on a batch."""
    targets = batch['targets']
    weights = batch['weights']
    logits, _ = self.model_fn(
        params,
        batch,
        mode=spec.ForwardPassMode.EVAL,
        model_state=None,
        rng=None,
        update_batch_norm=False)
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

  def _normalize_eval_metrics(
      self, num_examples: int, total_metrics: Dict[str,
                                                   Any]) -> Dict[str, float]:
    """Normalize eval metrics."""
    del num_examples
    if USE_PYTORCH_DDP:
      for metric in total_metrics.values():
        dist.all_reduce(metric)
    total_metrics = {k: v.item() for k, v in total_metrics.items()}
    eval_denominator = total_metrics.pop('denominator')
    return jax.tree_map(lambda x: float(x / eval_denominator), total_metrics)


class WmtWorkloadPostLN(WmtWorkload):
  """WMT PyTorch workload with post instead of pre layer norm."""

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
  """WMT PyTorch workload with attention temperature = 4.0."""

  @property
  def validation_target_value(self) -> float:
    return 29.3379

  @property
  def test_target_value(self) -> float:
    return 229.4143

  @property
  def attention_temp(self) -> float:
    return 4.0


class WmtWorkloadGLUTanH(WmtWorkload):
  """WMT PyTorch workload with GLU and TanH activations."""

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
