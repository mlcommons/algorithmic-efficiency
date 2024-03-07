"""Training algorithm track submission functions for MNIST."""

from typing import Dict, Iterator, List, Tuple

import torch

from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.lr_scheduler import SequentialLR

from algorithmic_efficiency import spec


def get_batch_size(workload_name):
  # Return the global batch size.
  batch_sizes = {'mnist': 1024}
  return batch_sizes[workload_name]


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del model_state
  del workload
  del rng
  optimizer_state = {
      'optimizer':
          torch.optim.Adam(
              model_params.parameters(),
              lr=hyperparameters.learning_rate,
              betas=(1.0 - hyperparameters.one_minus_beta_1, 0.999),
              eps=hyperparameters.epsilon),
  }

  def pytorch_trapezoid(step_hint: int, hyperparameters, optimizer):
    
    warmup_steps = int(hyperparameters.warmup_factor * step_hint)
    decay_start_step = int(hyperparameters.decay_factor * step_hint)
    constant_steps = decay_start_step - warmup_steps
    decay_steps = step_hint - decay_start_step
    
    warmup = LinearLR(
        optimizer, start_factor=1e-10, end_factor=1., total_iters=warmup_steps)
    constant = ConstantLR(
      optimizer, factor=1.0, total_iters=constant_steps)
    linear_decay = LinearLR(
        optimizer, start_factor=1., end_factor=0., total_iters=decay_steps)
    
    return SequentialLR(
        optimizer,
        schedulers=[warmup, constant, linear_decay],
        milestones=[warmup_steps, decay_start_step])

  optimizer_state['scheduler'] = pytorch_trapezoid(
   workload.step_hint, hyperparameters, optimizer_state['optimizer'])


  return optimizer_state


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
  del hyperparameters
  del loss_type
  del current_params_types
  del eval_results
  del global_step

  current_model = current_param_container
  current_model.train()
  for param in current_model.parameters():
    param.grad = None

  output, new_model_state = workload.model_fn(
      params=current_model,
      augmented_and_preprocessed_input_batch=batch,
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      update_batch_norm=True)

  loss_dict = workload.loss_fn(
      label_batch=batch['targets'], logits_batch=output)
  loss = loss_dict['summed'] / loss_dict['n_valid_examples']
  loss.backward()
  optimizer_state['optimizer'].step()


  import wandb
  if wandb.run is not None:
    wandb.log({
        'lr': optimizer_state['scheduler'].get_last_lr()[0], 
        'lr_step': global_step,
        })


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
  del optimizer_state
  del current_param_container
  del global_step
  del rng
  return next(input_queue)
