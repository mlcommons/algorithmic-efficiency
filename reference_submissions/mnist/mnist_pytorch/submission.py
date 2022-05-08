"""Training algorithm track submission functions for MNIST."""

from typing import Dict, Iterator, List, Tuple

import torch

from algorithmic_efficiency import spec

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_batch_size(workload_name):
  # Return the global batch size.
  batch_sizes = {'mnist': 1024}
  return batch_sizes[workload_name]


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del rng
  del model_state
  del workload

  optimizer_state = {
      'optimizer':
          torch.optim.Adam(
              model_params.parameters(), lr=hyperparameters.learning_rate)
  }
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
  del hyperparameters
  del loss_type
  del current_params_types
  del eval_results
  del global_step

  current_model = current_param_container
  current_param_container.train()
  optimizer_state['optimizer'].zero_grad()

  output, new_model_state = workload.model_fn(
      params=current_model,
      augmented_and_preprocessed_input_batch=batch['inputs'],
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      update_batch_norm=True)

  loss = workload.loss_fn(
      label_batch=batch['targets'], logits_batch=output).mean()

  loss.backward()
  optimizer_state['optimizer'].step()

  return (optimizer_state, current_param_container, new_model_state)


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Tuple[spec.Tensor, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Tuple[spec.Tensor, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a single training example and label.

  We left out `current_params_types` because we do not believe that it would
  # be necessary for this function.

  Return a tuple of input label batches.
  """
  del optimizer_state
  del current_param_container
  del global_step
  del rng
  return next(input_queue)
