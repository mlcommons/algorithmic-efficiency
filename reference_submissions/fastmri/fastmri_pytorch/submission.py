"""Training algorithm track submission functions for FastMRI."""

from typing import Dict, Iterator, List, Tuple

import torch
from torch.optim.lr_scheduler import StepLR

from algorithmic_efficiency import spec


def get_batch_size(workload_name):
  # Return the global batch size.
  batch_sizes = {'fastmri': 8}
  return batch_sizes[workload_name]


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del workload
  del model_state
  del rng

  base_lr = hyperparameters.learning_rate * get_batch_size('fastmri')
  optimizer_state = {
      'optimizer':
          torch.optim.RMSprop(
              model_params.parameters(),
              lr=base_lr,
              weight_decay=hyperparameters.l2)
  }

  optimizer_state['scheduler'] = StepLR(
      optimizer_state['optimizer'],
      step_size=hyperparameters.lr_step_size,
      gamma=hyperparameters.lr_gamma)

  return optimizer_state


def update_params(
    workload: spec.Workload,
    current_param_container: spec.ParameterContainer,
    current_params_types: spec.ParameterTypeTree,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparameters,
    batch: Dict[str, spec.Tensor],
    # This will define the output activation via `output_activation_fn`.
    loss_type: spec.LossType,
    optimizer_state: spec.OptimizerState,
    eval_results: List[Tuple[int, float]],
    global_step: int,
    rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params)."""
  del current_params_types
  del loss_type
  del eval_results

  current_model = current_param_container
  current_param_container.train()
  optimizer_state['optimizer'].zero_grad()
  if hasattr(hyperparameters, 'dropout_rate'):
    dropout_rate = hyperparameters.dropout_rate
  else:
    dropout_rate = 0.0

  outputs_batch, new_model_state = workload.model_fn(
      params=current_model,
      augmented_and_preprocessed_input_batch=batch,
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      dropout_rate=dropout_rate,
      aux_dropout_rate=None,
      update_batch_norm=True)

  loss = workload.loss_fn(
      label_batch=batch['targets'], logits_batch=outputs_batch).mean()

  loss.backward()
  optimizer_state['optimizer'].step()
  steps_per_epoch = workload.num_train_examples // get_batch_size('fastmri')
  if (global_step + 1) % steps_per_epoch == 0:
    optimizer_state['scheduler'].step()

  return (optimizer_state, current_param_container, new_model_state)


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
