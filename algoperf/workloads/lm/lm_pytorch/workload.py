"""LM workload implemented in PyTorch."""

from typing import Dict, Iterator, Optional, Tuple

import jax
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from algoperf import param_utils
from algoperf import pytorch_utils
from algoperf import spec
from algoperf.workloads.lm.workload import BaseLmWorkload
from algoperf.workloads.lm.lm_pytorch.models import LinearLayer

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()


class LmWorkload(BaseLmWorkload):
  """LM PyTorch workload."""

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    
    if hasattr(self, '_model'):
        self._model.reset_parameters()
        return self._model, None

    torch.manual_seed(rng[0])
    self._model = LinearLayer(vocab_size=self._vocab_size)
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
      batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    
    del model_state, rng, update_batch_norm  # Not used for linear model
    model = params
    inputs = batch['inputs'].float()  # Convert one-hot to float
    logits = model(inputs)
    return logits, None

  def _build_input_queue(
      self,
      data_rng: jax.random.PRNGKey,
      split: str,
      data_dir: str,
      global_batch_size: int,
      num_batches: Optional[int] = None,
      repeat_final_dataset: bool = False) -> Iterator[Dict[str, spec.Tensor]]:
    """Build an input queue for the given split."""
    from algoperf.workloads.lm.input_pipeline import get_hf_dataloader
    
    loader = get_hf_dataloader(
        cache_dir=data_dir,
        data_rng=data_rng,
        batch_size=global_batch_size,
        seq_len=self._seq_len,
        framework="torch",
        split=split)
    seq_len = self._seq_len 
    weights = None
    
    dtype = torch.long
    is_train = split == 'train'
    
    for batch in loader:
      inputs, targets = batch
      
      if USE_PYTORCH_DDP:
        if not is_train:
          # During eval, the batch size of the remainder might be different
          per_device_batch_size = torch.tensor(
              len(targets[0]), dtype=dtype, device=DEVICE)
          dist.broadcast(per_device_batch_size, src=0)
        
        # Broadcast to all devices
        dist.broadcast(inputs, src=0)
        dist.broadcast(targets, src=0)
      
      if weights is None:
        weights = torch.ones(inputs.shape[0], device=DEVICE)

      if weights is None:
        weights = torch.ones(per_device_batch_size, device=DEVICE)
      batch = {
          'inputs': inputs,
          'targets': targets,
          'weights': weights,
      }
      yield batch

  def is_output_params(self, param_name: str) -> bool:
    """Return whether the given parameter is an output parameter."""
    return 'output.weight' in param_name or 'output.bias' in param_name
    
  def _eval_batch(self,
                  params: spec.ParameterContainer,
                  batch: Dict[str, spec.Tensor],
                  model_state: spec.ModelAuxiliaryState,
                  rng: spec.RandomState) -> spec.Tensor:
    """Evaluate the model on a single batch."""
    model = params
    logits, _ = self.model_fn(
        model, batch, model_state, spec.ForwardPassMode.EVAL, rng, False)
    targets = batch['targets']
    
    # Calculate cross-entropy loss
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    loss = -torch.sum(targets * log_probs)
    return loss
  def loss_fn(
      self,
      label_batch: spec.Tensor,
      logits_batch: spec.Tensor,
      mask_batch: Optional[spec.Tensor] = None,
      label_smoothing: float = 0.0) -> Dict[str, spec.Tensor]:
    """Compute cross-entropy loss for language modeling in PyTorch."""
    vocab_size = logits_batch.shape[-1]
    
    if len(label_batch.shape) == len(logits_batch.shape):
      # One-hot labels
      log_probs = torch.nn.functional.log_softmax(logits_batch, dim=-1)
      loss = -torch.sum(label_batch * log_probs, dim=-1)
    else:
      # Dense labels
      loss = torch.nn.functional.cross_entropy(
          logits_batch, 
          label_batch,
          reduction='none')
    
    if mask_batch is not None:
      loss = loss * mask_batch
    
    n_valid = mask_batch.sum() if mask_batch is not None else label_batch.shape[0]
    return {
        'summed': loss.sum(),
        'n_valid_examples': n_valid,
        'per_example': loss
    }
