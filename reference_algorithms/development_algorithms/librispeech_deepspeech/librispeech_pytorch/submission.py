"""Training algorithm track submission functions for LibriSpeech."""
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.distributed.nn as dist_nn

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP = pytorch_setup()[0]


def get_batch_size(workload_name):
  # Return the global batch size.
  del workload_name
  return 256


def get_learning_rate(step, hyperparams):
  warmup_steps = hyperparams.warmup_steps
  if step < warmup_steps:
    current_lr = (step * hyperparams.base_lr) / warmup_steps
  else:
    decay_factor = (1 + np.cos(step / hyperparams.training_steps * np.pi)) * 0.5
    current_lr = hyperparams.base_lr * decay_factor
  return current_lr


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del workload
  del model_state
  del rng
  optimizer = torch.optim.AdamW(
      params=model_params.parameters(),
      lr=0.0,
      betas=(hyperparameters.beta1, hyperparameters.beta2),
      eps=hyperparameters.epsilon,
      weight_decay=hyperparameters.weight_decay)
  return {'optimizer': optimizer}


def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params)."""
  del current_params_types
  del eval_results
  del model_state
  del loss_type
  optimizer = optimizer_state['optimizer']
  optimizer.zero_grad()
  current_model = current_param_container

  (logits, logits_padding), _ = workload.model_fn(
      current_model,
      batch,
      None,
      spec.ForwardPassMode.TRAIN,
      rng,
      update_batch_norm=True)

  loss_dict = workload.loss_fn(batch['targets'], (logits, logits_padding))
  summed_loss = loss_dict['summed']
  n_valid_examples = loss_dict['n_valid_examples']
  if USE_PYTORCH_DDP:
    # Use dist_nn.all_reduce to ensure correct loss and gradient scaling.
    summed_loss = dist_nn.all_reduce(summed_loss)
    n_valid_examples = dist_nn.all_reduce(n_valid_examples)
  loss = summed_loss / n_valid_examples

  loss.backward()

  for g in optimizer.param_groups:
    g['lr'] = get_learning_rate(global_step, hyperparameters)
  if hasattr(hyperparameters, 'grad_clip'):
    torch.nn.utils.clip_grad_norm_(
        current_model.parameters(), max_norm=hyperparameters.grad_clip)
  optimizer.step()
  return optimizer_state, current_param_container, None


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   model_state: spec.ModelAuxiliaryState,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

    Each element of the queue is a batch of training examples and labels.
  """
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del global_step
  del rng
  return next(input_queue)
