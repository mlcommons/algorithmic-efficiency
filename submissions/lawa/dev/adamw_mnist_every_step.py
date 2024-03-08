"""Training algorithm track submission functions for MNIST."""

from typing import Dict, Iterator, List, Tuple

from absl import logging
import torch
import wandb

from copy import deepcopy

import math

import pdb
from algorithmic_efficiency import spec
from collections import deque

def get_batch_size(workload_name):
  # Return the global batch size.
  batch_sizes = {'mnist': 1024}
  return batch_sizes[workload_name]

class LAWAQueue:
  def __init__(self, maxlen) -> None:
    self._maxlen = int(maxlen)
    self._queue = deque(maxlen=self._maxlen)
  
  def state_dict(self):
    return {key: value for key, value in self.__dict__.items()}
  
  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)
    
  def push(self, params):
    self._queue.append([p.detach().clone(memory_format=torch.preserve_format) for p in params])
  
  def get_last(self):
    return self._queue[-1]
  
  def full(self):
    return (len(self._queue)==self._maxlen)

  def get_avg(self):
    if not self.full():
      raise ValueError("q should be full to compute avg")
    
    q = self._queue
    k = float(self._maxlen)
    q_avg = [torch.zeros_like(p, device=p.device) for p in q[0]]
    for chkpts in q:
      for p_avg,p in zip(q_avg, chkpts):
        p_avg.add_(p/k)
    
    return q_avg

def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del model_state
  # del workload
  del rng
  optimizer_state = {
      'optimizer':
          torch.optim.Adam(
              model_params.parameters(),
              lr=hyperparameters.learning_rate,
              betas=(1.0 - hyperparameters.one_minus_beta_1, 0.999),
              eps=hyperparameters.epsilon),
        'queue': LAWAQueue(maxlen=hyperparameters.k),
  }
  return optimizer_state


def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState, # sempre None da workload.model_fn()
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params)."""
  del loss_type
  del current_params_types
  del eval_results
  
  current_model = current_param_container
  queue = optimizer_state['queue']
  
  # ### check that at new iteration, params are same as avg
  # if wandb.run is not None and queue.full():
  #   def mynorm(params):
  #     return torch.norm(torch.stack([torch.norm(p.detach().clone(), 2) for p in params]), 2)
  #   wandb.log({
  #       'norm_model': mynorm(current_model.parameters()),
  #       'norm_avg': mynorm(queue.get_avg()),
  #   })
    
  # Discard average and load previous params after loading
  if queue.full():
    for p,p_old in zip(current_model.parameters(), 
                       queue.get_last()):
      p.data = p_old.clone()
  
  # ### check that model params equal prev_params
  # if wandb.run is not None and queue.full():
  #   def mynorm(params):
  #     return torch.norm(torch.stack([torch.norm(p.detach().clone(), 2) for p in params]), 2)
  #   wandb.log({
  #       'norm_model': mynorm(current_model.parameters()),
  #       'norm_prev': mynorm(queue.get_last()),
  #   })
    
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
  
  # Update queue
  queue.push(current_model.parameters())
  
  if queue.full():
    # Compute avg
    avg = queue.get_avg()
    # Load avg into model
    for p, p_avg in zip(current_model.parameters(), avg):
      assert p.data.shape == p_avg.shape, "Shape mismatch"
      p.data = p_avg.clone()
    
    # ### check that Load avg into model does not affect previous_state_dict
    # if wandb.run is not None:
    #   def mynorm(params):
    #     return torch.norm(torch.stack([torch.norm(p.detach().clone(), 2) for p in params]), 2)
    #   wandb.log({
    #       'norm_avg': mynorm(avg),
    #       'norm_model': mynorm(current_model.parameters()),
    #       'norm_prev': mynorm(queue.get_last()),
    #   })
  
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
