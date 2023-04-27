"""Submission file for a SGD with Nesterov optimizer in PyTorch."""

import torch
from torch.optim.lr_scheduler import LambdaLR

from algorithmic_efficiency import spec
from reference_algorithms.target_setting_algorithms.data_selection import \
    data_selection  # pylint: disable=unused-import
from reference_algorithms.target_setting_algorithms.get_batch_size import \
    get_batch_size  # pylint: disable=unused-import
from reference_algorithms.target_setting_algorithms.jax_nesterov import \
    create_lr_schedule_fn
from reference_algorithms.target_setting_algorithms.pytorch_submission_base import \
    update_params  # pylint: disable=unused-import


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a Nesterov optimizer and a learning rate schedule."""
  del model_state
  del rng

  # Create optimizer.
  optimizer_state = {
      'optimizer':
          torch.optim.SGD(
              model_params.parameters(),
              lr=hyperparameters.learning_rate,
              momentum=hyperparameters.beta1,
              weight_decay=hyperparameters.weight_decay,
              nesterov=True),
  }

  # Create learning rate schedule.
  target_setting_step_hint = int(0.75 * workload.step_hint)
  lr_schedule_fn = create_lr_schedule_fn(target_setting_step_hint,
                                         hyperparameters)

  # PyTorch's LambdaLR expects the lr_lambda fn to return a factor which will
  # be multiplied with the base lr, so we have to divide by it here.
  def _lr_lambda(step: int) -> float:
    return lr_schedule_fn(step).item() / hyperparameters.learning_rate

  optimizer_state['scheduler'] = LambdaLR(
      optimizer_state['optimizer'], lr_lambda=_lr_lambda)

  return optimizer_state
