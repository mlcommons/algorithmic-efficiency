"""Training algorithm track submission functions for ImageNet."""
from typing import Dict, Iterator, List, Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

from algorithmic_efficiency import spec


def get_batch_size(workload_name):
  # Return the global batch size.
  del workload_name
  return 512


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del model_state
  del rng

  batch_size = get_batch_size('imagenet_resnet')
  base_lr = hyperparameters.learning_rate * batch_size / 256.
  optimizer_state = {
      'optimizer':
          torch.optim.SGD(
              model_params.parameters(),
              lr=base_lr,
              momentum=hyperparameters.momentum,
              weight_decay=hyperparameters.l2,
              nesterov=True)
  }

  steps_per_epoch = workload.num_train_examples // batch_size
  scheduler1 = LinearLR(
      optimizer_state['optimizer'],
      start_factor=1e-10,
      end_factor=1.,
      total_iters=hyperparameters.warmup_epochs * steps_per_epoch)
  cosine_epochs = max(
      hyperparameters.num_epochs - hyperparameters.warmup_epochs, 1)
  scheduler2 = CosineAnnealingLR(
      optimizer_state['optimizer'], T_max=cosine_epochs * steps_per_epoch)

  optimizer_state['scheduler'] = SequentialLR(
      optimizer_state['optimizer'],
      schedulers=[scheduler1, scheduler2],
      milestones=[hyperparameters.warmup_epochs * steps_per_epoch])

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
  del hyperparameters
  del loss_type
  del eval_results
  del global_step

  current_model = current_param_container
  current_param_container.train()
  optimizer_state['optimizer'].zero_grad()

  logits_batch, new_model_state = workload.model_fn(
      params=current_model,
      augmented_and_preprocessed_input_batch=batch,
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      update_batch_norm=True)

  loss = workload.loss_fn(
      label_batch=batch['targets'], logits_batch=logits_batch).mean()

  loss.backward()
  optimizer_state['optimizer'].step()
  optimizer_state['scheduler'].step()

  return (optimizer_state, current_param_container, new_model_state)


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a batch of training examples and labels.
  """
  del workload
  del optimizer_state
  del current_param_container
  del hyperparameters
  del global_step
  del rng
  return next(input_queue)
