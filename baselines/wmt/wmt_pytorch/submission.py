from typing import Iterator, List, Tuple
import numpy as np
import torch

from algorithmic_efficiency import spec

def get_batch_size(workload_name):
  batch_sizes = {'wmt_pytorch': 128}
  return batch_sizes[workload_name]


def create_learning_rate_scheduler(
    factors="constant * linear_warmup * rsqrt_decay",
    base_learning_rate=0.5,
    warmup_steps=1000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000):
  """Creates learning rate schedule.
  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.
  Args:
    factors: string, factors separated by "*" that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: int, how many steps to warm up for in the warmup schedule.
    decay_factor: float, the amount to decay the learning rate by.
    steps_per_decay: int, how often to decay the learning rate.
    steps_per_cycle: int, steps per cycle when using cosine decay.
  Returns:
    a function learning_rate(step): float -> {"learning_rate": float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split("*")]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == "constant":
        ret *= base_learning_rate
      elif name == "linear_warmup":
        ret *= np.minimum(1.0, step / warmup_steps)
      elif name == "rsqrt_decay":
        ret /= np.sqrt(np.maximum(step, warmup_steps))
      elif name == "rsqrt_normalized_decay":
        ret *= np.sqrt(warmup_steps)
        ret /= np.sqrt(np.maximum(step, warmup_steps))
      elif name == "decay_every":
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == "cosine_decay":
        progress = np.maximum(
            0.0, (step - warmup_steps) / float(steps_per_cycle))
        ret *= np.maximum(
            0.0, 0.5 * (1.0 + np.cos(np.pi * (progress % 1.0))))
      else:
        raise ValueError("Unknown factor %s." % name)
    return ret

  return step_fn


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparamters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del workload
  del model_state
  del rng

  optimizer_state = {
      'optimizer':
          torch.optim.Adam(
              model_params.parameters(),
              lr=hyperparameters.learning_rate,
              betas=(1.0 - hyperparameters.one_minus_beta_1, 0.98),
              eps=hyperparameters.epsilon)
  }

  optimizer_state['scheduler'] = create_learning_rate_scheduler(
      base_learning_rate=hyperparameters.learning_rate)
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
  del eval_results
  del loss_type
  del hyperparameters
  del label_batch
  
  input_batch = workload.preprocess_for_train(
      input_batch, None, None, None, None, packed_examples=True)

  current_model = current_param_container
  current_param_container.train()
  optimizer = optimizer_state['optimizer']
  optimizer.zero_grad()

  output, new_model_state = workload.model_fn(
      params=current_model,
      input_batch=input_batch,
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      update_batch_norm=True)

  targets = input_batch[1]
  weights = torch.where(targets > 0, 1.0, 0.0)
  loss = (workload.loss_fn(targets, output) * weights).sum() / weights.sum()
  loss.backward()

  lr = optimizer_state['scheduler'](global_step).item()
  for g in optimizer.param_groups:
    g['lr'] = lr
  optimizer.step()

  return (optimizer_state, current_param_container, new_model_state)


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Tuple[spec.Tensor, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   hyperparameters: spec.Hyperparamters, global_step: int,
                   rng: spec.RandomState) -> Tuple[spec.Tensor, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a single training example and label.

  We left out `current_params_types` because we do not believe that it would
  # be necessary for this function.

  Return a tuple of input label batches.
  """
  del workload
  del optimizer_state
  del current_param_container
  del hyperparameters
  del global_step
  del rng
  return next(input_queue), None
