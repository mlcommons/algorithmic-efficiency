"""Submission file for an AdamW optimizer with warmup+cosine LR in PyTorch."""

import torch

from algoperf import spec
from algorithms.target_setting_algorithms import cosine_warmup
from algorithms.target_setting_algorithms.data_selection import (  # noqa: F401
  data_selection,
)
from algorithms.target_setting_algorithms.get_batch_size import (  # noqa: F401
  get_batch_size,
)
from algorithms.target_setting_algorithms.pytorch_submission_base import (  # noqa: F401
  update_params,
)


def init_optimizer_state(
  workload: spec.Workload,
  model_params: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  rng: spec.RandomState,
) -> spec.OptimizerState:
  """Creates an AdamW optimizer and a learning rate schedule."""
  del model_state
  del rng

  epsilon = (
    hyperparameters.epsilon if hasattr(hyperparameters, 'epsilon') else 1e-8
  )
  optimizer_state = {
    'optimizer': torch.optim.AdamW(
      model_params.parameters(),
      lr=hyperparameters.learning_rate,
      betas=(hyperparameters.beta1, hyperparameters.beta2),
      eps=epsilon,
      weight_decay=hyperparameters.weight_decay,
    ),
  }

  target_setting_step_hint = int(0.75 * workload.step_hint)
  optimizer_state['scheduler'] = cosine_warmup.pytorch_cosine_warmup(
    target_setting_step_hint, hyperparameters, optimizer_state['optimizer']
  )

  return optimizer_state
