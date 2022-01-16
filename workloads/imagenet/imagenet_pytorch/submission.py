"""Training algorithm track submission functions for ImageNet."""
from typing import List, Tuple
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import spec
from workloads.mnist.mnist_pytorch.submission import data_selection


def get_batch_size(workload_name):
  batch_sizes = {'imagenet_pytorch': 128}
  return batch_sizes[workload_name]


def init_optimizer_state(
    workload: spec.Workload,
    model_params: spec.ParameterContainer,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparamters,
    rng: spec.RandomState) -> spec.OptimizerState:
  del workload
  del model_state
  del rng

  base_lr = hyperparameters.learning_rate * get_batch_size('imagenet_pytorch') / 256.
  optimizer_state = {
      'optimizer': torch.optim.SGD(model_params.parameters(),
                                   lr=base_lr,
                                   momentum=hyperparameters.momentum,
                                   weight_decay=hyperparameters.weight_decay)
  }

  scheduler1 = LinearLR(
      optimizer_state['optimizer'], start_factor=1e-5, end_factor=1.,
      total_iters=hyperparameters.warmup_epochs)
  cosine_epochs = max(
      hyperparameters.num_epochs - hyperparameters.warmup_epochs, 1)
  scheduler2 = CosineAnnealingLR(
      optimizer_state['optimizer'], T_max=cosine_epochs)

  optimizer_state['scheduler'] = SequentialLR(
      optimizer_state['optimizer'], schedulers=[scheduler1, scheduler2], 
      milestones=[hyperparameters.warmup_epochs])

  return optimizer_state


def update_params(
    workload: spec.Workload,
    current_param_container: spec.ParameterContainer,
    current_params_types: spec.ParameterTypeTree,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparamters,
    input_batch: spec.Tensor,
    label_batch: spec.Tensor,
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
  input_batch, label_batch = (
    workload.preprocess_for_train(input_batch, label_batch, None, None, None))

  current_model = current_param_container
  current_param_container.train()
  optimizer_state['optimizer'].zero_grad()

  output, new_model_state = workload.model_fn(
      params=current_model,
      input_batch=input_batch,
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      update_batch_norm=True)

  loss = workload.loss_fn(
      label_batch=label_batch,
      logits_batch=output)

  loss.backward()
  optimizer_state['optimizer'].step()

  steps_per_epoch = workload.num_train_examples // get_batch_size('imagenet_pytorch')
  if (global_step + 1) % steps_per_epoch == 0:
    optimizer_state['scheduler'].step()

  return (optimizer_state, current_param_container, new_model_state)
