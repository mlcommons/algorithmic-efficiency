"""Submission file for a SAM optimizer with warmup+cosine LR in PyTorch."""

from typing import Callable, Dict, Iterator, List, Tuple

from absl import logging
import torch
import torch.distributed.nn as dist_nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP = pytorch_setup()[0]


# Modified from https://github.com/davda54/sam.
class SAM(torch.optim.Optimizer):

  def __init__(self,
               params: spec.ParameterContainer,
               base_optimizer: torch.optim.Optimizer,
               rho: float = 0.05,
               adaptive: bool = False,
               **kwargs):
    if rho < 0.0:
      raise ValueError(f'Invalid rho, should be non-negative: {rho}')

    defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
    super().__init__(params, defaults)

    self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
    self.param_groups = self.base_optimizer.param_groups
    self.defaults.update(self.base_optimizer.defaults)

  @torch.no_grad()
  def first_step(self, zero_grad: bool = False):
    grad_norm = self._grad_norm()
    for group in self.param_groups:
      scale = group['rho'] / grad_norm

      for p in group['params']:
        if p.grad is None:
          continue
        self.state[p]['old_p'] = p.data.clone()
        factor = torch.pow(p, 2) if group['adaptive'] else 1.0
        e_w = factor * p.grad * scale.to(p)
        p.add_(e_w)  # Climb to the local maximum 'w + e(w)'.

    if zero_grad:
      self.zero_grad()

  @torch.no_grad()
  def second_step(self, zero_grad: bool = False):
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        p.data = self.state[p]['old_p']  # Get back to 'w' from 'w + e(w)'.

    self.base_optimizer.step()  # Do the actual 'sharpness-aware' update.

    if zero_grad:
      self.zero_grad()

  @torch.no_grad()
  def step(self, closure: Callable = None):
    if closure is None:
      raise ValueError('SAM requires closure, but it was not provided.')
    # The closure should do a full forward-backward pass.
    closure = torch.enable_grad()(closure)

    self.first_step(zero_grad=True)
    closure()
    self.second_step()

  def _grad_norm(self):
    # In case of model parallelism, put everything on the same device.
    shared_device = self.param_groups[0]['params'][0].device
    norm = torch.norm(
        torch.stack([((torch.abs(p) if group['adaptive'] else 1.0) *
                      p.grad).norm(p=2).to(shared_device)
                     for group in self.param_groups
                     for p in group['params']
                     if p.grad is not None]),
        p=2)
    return norm

  def load_state_dict(self, state_dict: Dict):
    super().load_state_dict(state_dict)
    self.base_optimizer.param_groups = self.param_groups


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates an AdamW optimizer and a learning rate schedule."""
  del model_state
  del rng

  # Create SAM optimizer with AdamW base.
  base_optimizer = torch.optim.AdamW
  optimizer_state = {
      'optimizer':
          SAM(model_params.parameters(),
              base_optimizer=base_optimizer,
              rho=hyperparameters.rho,
              lr=hyperparameters.learning_rate,
              betas=(1.0 - hyperparameters.one_minus_beta1,
                     hyperparameters.beta2),
              eps=1e-8,
              weight_decay=hyperparameters.weight_decay),
  }

  def pytorch_cosine_warmup(step_hint: int, hyperparameters, optimizer):
    warmup_steps = int(hyperparameters.warmup_factor * step_hint)
    warmup = LinearLR(
        optimizer, start_factor=1e-10, end_factor=1., total_iters=warmup_steps)
    cosine_steps = max(step_hint - warmup_steps, 1)
    cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine_decay], milestones=[warmup_steps])

  # Create learning rate schedule.
  optimizer_state['scheduler'] = pytorch_cosine_warmup(
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
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del eval_results

  current_model = current_param_container
  current_model.train()

  def _loss_fn(params, update_batch_norm=True):
    """Loss function used for training."""
    logits_batch, new_model_state = workload.model_fn(
        params=params,
        augmented_and_preprocessed_input_batch=batch,
        model_state=model_state,
        mode=spec.ForwardPassMode.TRAIN,
        rng=rng,
        update_batch_norm=update_batch_norm)
    label_smoothing = (
        hyperparameters.label_smoothing if hasattr(hyperparameters,
                                                   'label_smoothing') else 0.0)
    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits_batch,
        mask_batch=batch.get('weights'),
        label_smoothing=label_smoothing)
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    if USE_PYTORCH_DDP:
      # Use dist_nn.all_reduce to ensure correct loss and gradient scaling.
      summed_loss = dist_nn.all_reduce(summed_loss)
      n_valid_examples = dist_nn.all_reduce(n_valid_examples)
    loss = summed_loss / n_valid_examples
    return loss, new_model_state

  # First backward pass.
  loss, _ = _loss_fn(current_model, update_batch_norm=True)
  loss.backward()

  logging_loss = loss.clone().detach()
  with torch.no_grad():
    parameters = [p for p in current_model.parameters() if p.grad is not None]
    grad_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)

  optimizer_state['optimizer'].first_step(zero_grad=True)

  # Second forward-backward pass.
  loss, new_model_state = _loss_fn(current_model, update_batch_norm=False)
  loss.backward()

  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
    torch.nn.utils.clip_grad_norm_(
        current_model.parameters(), max_norm=grad_clip)
  optimizer_state['optimizer'].second_step(zero_grad=True)
  optimizer_state['scheduler'].step()

  # Log training metrics - loss, grad_norm, batch_size.
  if global_step <= 100 or global_step % 500 == 0:
    if workload.metrics_logger is not None:
      workload.metrics_logger.append_scalar_metrics(
          {
              'loss': logging_loss.item(),
              'grad_norm': grad_norm.item(),
          },
          global_step)
    logging.info('%d) loss = %0.3f, grad_norm = %0.3f',
                 global_step,
                 logging_loss.item(),
                 grad_norm.item())

  return (optimizer_state, current_param_container, new_model_state)


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262_144
  elif workoad_name == 'criteo1tb_layernorm'
    return 262_144
  elif workload_name == 'criteo1tb_resnet'
    return 262_144
  elif workload_name == 'fastmri':
    return 32
  elif workload_name == 'imagenet_resnet':
    return 1024
  elif workload_name == 'imagenet_vit':
    return 1024
  elif workload_name == 'librispeech_conformer':
    return 256
  elif workload_name == 'librispeech_deepspeech':
    return 256
  elif workload_name == 'ogbg':
    return 512
  elif workload_name == 'wmt':
    return 128
  elif workload_name == 'mnist':
    return 16
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')


def data_selection(
    workload: spec.Workload,
    input_queue: Iterator[Dict[str, spec.Tensor]],
    optimizer_state: spec.OptimizerState,
    current_param_container: spec.ParameterContainer,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparameters,
    global_step: int,
    rng: spec.RandomState) -> Tuple[spec.Tensor, spec.Tensor, spec.Tensor]:
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
  batch = next(input_queue)
  return batch
