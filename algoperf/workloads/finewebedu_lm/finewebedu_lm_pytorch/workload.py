"""LM workload implemented in PyTorch."""

import contextlib
from itertools import islice
from typing import Any, Dict, Iterator, Optional, Tuple

import jax
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from algoperf import param_utils, pytorch_utils, spec
from algoperf.workloads.finewebedu_lm.finewebedu_lm_pytorch.models import (
  ModelConfig,
  Transformer,
)
from algoperf.workloads.finewebedu_lm.input_pipeline import get_data_iter
from algoperf.workloads.finewebedu_lm.workload import BaseLmWorkload

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()

# Dtype mapping from string to PyTorch dtype
DTYPE_MAP = {
  'float32': torch.float32,
  'float16': torch.float16,
  'bfloat16': torch.bfloat16,
}


class LmWorkload(BaseLmWorkload):
  """LM PyTorch workload."""

  @property
  def _compute_dtype(self) -> torch.dtype:
    return DTYPE_MAP[self._compute_dtype_str]

  @property
  def _param_dtype(self) -> torch.dtype:
    return DTYPE_MAP[self._param_dtype_str]

  def init_model_fn(
    self,
    rng: spec.RandomState,
    dropout_rate: Optional[float] = None,
    aux_dropout_rate: Optional[float] = None,
  ) -> spec.ModelInitState:
    if hasattr(self, '_model'):
      # Reinitialize weights but keep same config
      self._model.apply(self._model._init_weights)
      self._model._scale_residual_branches()
      return self._model, None

    torch.manual_seed(rng[0])
    cfg = ModelConfig(
      vocab_size=self._vocab_size,
      seq_len=self._seq_len,
      model_dim=self._emb_dim,  # Model dimension
      expanded_model_dim=self._mlp_dim,  # MLP expanded dim
      num_layers=self._n_layers,
      num_heads=self._n_heads,
      rmsnorm_epsilon=self._rmsnorm_epsilon,
      qknorm_epsilon=self._qknorm_epsilon,
      tie_embeddings=self._tie_embeddings,
      compute_dtype=self._compute_dtype,
      param_dtype=self._param_dtype,
    )
    self._model = Transformer(cfg)
    self._param_shapes = param_utils.pytorch_param_shapes(self._model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    self._model.to(DEVICE)

    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        self._model = DDP(self._model, device_ids=[RANK], output_device=RANK)
      else:
        self._model = torch.nn.DataParallel(self._model)

    return self._model, None

  def model_fn(
    self,
    params: spec.ParameterContainer,
    augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
    model_state: spec.ModelAuxiliaryState,
    mode: spec.ForwardPassMode,
    rng: spec.RandomState,
    update_batch_norm: bool,
    dropout_rate: float = 0.0,
  ) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state, rng, update_batch_norm, dropout_rate
    model = params

    # Set model to eval or train mode based on the mode parameter
    if mode == spec.ForwardPassMode.EVAL:
      model.eval()
    elif mode == spec.ForwardPassMode.TRAIN:
      model.train()
    contexts = {
      spec.ForwardPassMode.EVAL: torch.no_grad,
      spec.ForwardPassMode.TRAIN: contextlib.nullcontext,
    }

    # Determine device type for autocast
    device_type = 'cuda' if DEVICE.type == 'cuda' else 'cpu'

    with contexts[mode]():
      with torch.autocast(device_type=device_type, dtype=self._compute_dtype):
        # Convert one-hot inputs to token IDs if needed
        inputs = augmented_and_preprocessed_input_batch['inputs']
        if inputs.dim() == 3:  # one-hot encoded
          inputs = inputs.argmax(dim=-1)

        logits = model(inputs)

    return logits, None

  def _build_input_queue(
    self,
    data_rng: jax.random.PRNGKey,
    split: str,
    data_dir: str,
    global_batch_size: int,
    cache: Optional[bool] = None,
    repeat_final_dataset: Optional[bool] = None,
    num_batches: Optional[int] = None,
  ) -> Iterator[Dict[str, spec.Tensor]]:
    """Build an input queue for the given split."""
    del cache, repeat_final_dataset
    local_batch_size = global_batch_size // N_GPUS
    loader = get_data_iter(
      data_rng=data_rng,
      split=split,
      data_dir=data_dir,
      batch_size=local_batch_size,
      num_batches=num_batches,
    )
    if USE_PYTORCH_DDP:
      loader = islice(loader, RANK, None, N_GPUS)
    dtype = torch.int32
    for batch in loader:
      batch = {
        'inputs': torch.tensor(batch['inputs'], device=DEVICE, dtype=dtype),
        'targets': torch.tensor(
          batch['targets'], device=DEVICE, dtype=torch.int64
        ),
        'weights': torch.tensor(
          batch['weights'], device=DEVICE, dtype=self._param_dtype
        )
        if batch['weights'] is not None
        else None,
      }
      yield batch

  def is_output_params(self, param_name: str) -> bool:
    """Return whether the given parameter is an output parameter."""
    return 'lm_head.weight' in param_name or 'lm_head.bias' in param_name

  def loss_fn(
    self,
    label_batch: spec.Tensor,
    logits_batch: spec.Tensor,
    mask_batch: spec.Tensor,
    label_smoothing: float = 0.0,
  ) -> Dict[str, spec.Tensor]:
    """Compute weighted cross-entropy loss.

    Args:
      label_batch: Target labels of shape [batch, length] (int).
      logits_batch: Predicted logits of shape [batch, length, vocab_size] (float).
      mask_batch: Optional weights of shape [batch, length] (float). Used to mask
        out padding tokens or weight examples differently. If None, all examples
        are weighted equally.
      label_smoothing: Label smoothing factor in [0, 1]. When > 0, the target
        distribution becomes (1 - label_smoothing) for the correct class and
        label_smoothing / vocab_size for all other classes. Default is 0.0 (no smoothing).

    Returns:
      Dictionary containing:
        - 'summed': Scalar tensor with the sum of all weighted losses.
        - 'n_valid_examples': Scalar tensor with the count of valid (non-masked) examples.
        - 'per_example': Tensor of shape [batch, length] with individual losses per example.
    """
    # Determine device type for autocast
    device_type = 'cuda' if logits_batch.is_cuda else 'cpu'

    with torch.autocast(device_type=device_type, dtype=self._compute_dtype):
      vocab_size = logits_batch.size(-1)

      # Compute cross-entropy loss with label smoothing
      per_example_losses = torch.nn.functional.cross_entropy(
        logits_batch.view(-1, vocab_size),
        label_batch.view(-1),
        reduction='none',
        label_smoothing=label_smoothing,
      )
      per_example_losses = per_example_losses.view_as(label_batch)

      # Apply weights if provided
      if mask_batch is not None:
        per_example_losses = per_example_losses * mask_batch

      # Calculate number of valid examples
      n_valid_examples = (
        mask_batch.sum()
        if mask_batch is not None
        else torch.tensor(
          label_batch.numel(),
          dtype=self._param_dtype,
          device=label_batch.device,
        )
      )

    return {
      'summed': per_example_losses.sum(),
      'n_valid_examples': n_valid_examples,
      'per_example': per_example_losses,
    }

  def _eval_batch(
    self,
    params: spec.ParameterContainer,
    batch: Dict[str, spec.Tensor],
    model_state: spec.ModelAuxiliaryState,
    rng: spec.RandomState,
  ) -> spec.Tensor:
    """Evaluate the model on a single batch."""
    logits, _ = self.model_fn(
      params, batch, model_state, spec.ForwardPassMode.EVAL, rng, False
    )
    metrics = self.loss_fn(
      label_batch=batch['targets'],
      logits_batch=logits,
      mask_batch=batch['weights'],
    )
    return {
      'loss': metrics['summed'].detach(),
      'denominator': metrics['n_valid_examples'].detach(),
    }

  def _normalize_eval_metrics(
    self, num_examples: int, total_metrics: Dict[str, Any]
  ) -> Dict[str, float]:
    """Normalize eval metrics."""
    del num_examples
    if USE_PYTORCH_DDP:
      for metric in total_metrics.values():
        dist.all_reduce(metric)
    total_metrics = {k: v.item() for k, v in total_metrics.items()}
    eval_denominator = total_metrics.pop('denominator')
    return jax.tree.map(lambda x: float(x / eval_denominator), total_metrics)
