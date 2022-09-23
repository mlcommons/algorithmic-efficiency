"""Submission file for an AdamW optimizer in PyTorch."""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

from algorithmic_efficiency import spec


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates an AdamW optimizer and a learning rate schedule."""
  del workload
  del model_state
  del rng

  epsilon = (
      hyperparameters.epsilon if hasattr(hyperparameters, 'epsilon') else 1e-8)
  optimizer_state = {
      'optimizer':
          torch.optim.AdamW(
              model_params.parameters(),
              lr=hyperparameters.learning_rate,
              betas=(hyperparameters.beta1, hyperparameters.beta2),
              eps=epsilon,
              weight_decay=hyperparameters.l2)
  }

  scheduler1 = LinearLR(
      optimizer_state['optimizer'],
      start_factor=1e-10,
      end_factor=1.,
      total_iters=hyperparameters.warmup_steps)
  cosine_steps = max(hyperparameters.num_steps - hyperparameters.warmup_steps,
                     1)
  scheduler2 = CosineAnnealingLR(
      optimizer_state['optimizer'], T_max=cosine_steps)

  optimizer_state['scheduler'] = SequentialLR(
      optimizer_state['optimizer'],
      schedulers=[scheduler1, scheduler2],
      milestones=[hyperparameters.warmup_steps])

  return optimizer_state
