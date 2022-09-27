"""PyTorch submission for the target-setting run on OGBG with AdamW."""

from typing import Dict, List, Tuple

import torch

from algorithmic_efficiency import spec
from target_setting_runs.data_selection import \
    data_selection  # pylint: disable=unused-import
from target_setting_runs.pytorch_nadamw import \
    init_optimizer_state  # pylint: disable=unused-import


def get_batch_size(workload_name):
  # Return the global batch size.
  del workload_name
  return 512


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
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del eval_results
  del global_step

  current_model = current_param_container
  current_model.train()
  optimizer_state['optimizer'].zero_grad()

  logits, new_model_state = workload.model_fn(
      params=current_model,
      augmented_and_preprocessed_input_batch=batch,
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      dropout_prob=0.1,  # Default.
      aux_dropout_prob=None,
      update_batch_norm=True)

  mask = batch['weights']
  label_smoothing = (
      hyperparameters.label_smoothing if hasattr(hyperparameters,
                                                 'label_smoothing') else 0.0)
  per_example_losses = workload.loss_fn(
      batch['targets'], logits, mask, label_smoothing=label_smoothing)
  loss = torch.where(mask, per_example_losses, 0).sum() / mask.sum()

  loss.backward()
  optimizer_state['optimizer'].step()
  optimizer_state['scheduler'].step()

  return optimizer_state, current_param_container, new_model_state
