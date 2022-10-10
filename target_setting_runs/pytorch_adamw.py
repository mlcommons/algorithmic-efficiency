"""Submission file for an AdamW optimizer with warmup+cosine LR in PyTorch."""

import torch

from algorithmic_efficiency import spec
from target_setting_runs import cosine_warmup
from target_setting_runs.data_selection import \
    data_selection  # pylint: disable=unused-import
from target_setting_runs.get_batch_size import \
    get_batch_size  # pylint: disable=unused-import
from target_setting_runs.pytorch_submission_base import \
    update_params  # pylint: disable=unused-import


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

  optimizer_state['scheduler'] = cosine_warmup.pytorch_cosine_warmup(
      workload.step_hint, hyperparameters, optimizer_state['optimizer'])

  return optimizer_state
