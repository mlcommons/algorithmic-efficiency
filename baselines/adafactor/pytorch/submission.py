"""Submission file for Adafactor in PyTorch."""

from functools import partial
from typing import Dict, Iterator, List, Tuple

from absl import logging
import torch
import torch.distributed.nn as dist_nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP = pytorch_setup()[0]


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates an Adafactor optimizer and a learning rate schedule."""
  del model_state
  del rng

  # Create optimizer.
  optimizer_state = {
      'optimizer':
          Adafactor(
              model_params.parameters(),
              lr=hyperparameters.learning_rate,
              beta1=1 - hyperparameters.one_minus_beta1,
              weight_decay=hyperparameters.weight_decay),
  }
  optimizer = optimizer_state['optimizer']
  warmup = LinearLR(
      optimizer,
      start_factor=1e-10,
      end_factor=1.,
      total_iters=hyperparameters.warmup_steps)
  cosine_steps = max(workload.step_hint - hyperparameters.warmup_steps, 1)
  cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps)
  optimizer_state['scheduler'] = SequentialLR(
      optimizer,
      schedulers=[warmup, cosine_decay],
      milestones=[hyperparameters.warmup_steps])
  return optimizer_state


class Adafactor(torch.optim.Optimizer):
  """Adapted from https://github.com/huggingface/transformers/blob/main/
  src/transformers/optimization.py#L386"""

  def __init__(
      self,
      params,
      lr=None,
      beta1=0.9,
      decay_adam=0.99,
      weight_decay=0.0,
  ):
    defaults = dict(
        lr=lr,
        beta1=beta1,
        decay_adam=decay_adam,
        weight_decay=weight_decay,
        decay_pow=0.0,
        layerwise_adaptation=False,
        decay_method='adam',
        clip_threshold=1.0,
        factored=True,
        epsilon1_grad_sq_reg=1e-30,
        respect_skip_lp_regularization=False,
        exclude_from_layerwise_adaptation=None,
        per_var_learning_summary=False,
        sort_factored_second_moment_dims=False,
        # Unused because sort_factored_second_moment_dims=False.
        min_dim_size_to_factor=128,
        multiply_by_parameter_scale=False,
        # Unused because multiply_by_parameter_scale=False.
        epsilon2_param_scale_reg=1e-3,
        maybe_inf_to_nan=True,
    )
    super().__init__(params, defaults)

  def inf_to_nan(self, group, x):
    if group["maybe_inf_to_nan"]:
      x = torch.nan_to_num(x, nan=torch.nan, posinf=torch.nan, neginf=torch.nan)
    return x

  def step(self, closure=None):
    """
        Performs a single optimization step
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      inf_to_nan = partial(self.inf_to_nan, group)
      for p in group["params"]:
        if p.grad is None:
          continue
        grad = p.grad.data
        grad = inf_to_nan(grad)
        if grad.dtype in {torch.float16, torch.bfloat16}:
          grad = grad.float()
        if grad.is_sparse:
          raise RuntimeError("Adafactor does not support sparse gradients.")

        state = self.state[p]
        grad_shape = grad.shape

        factored = len(grad_shape) >= 2

        # State Initialization
        if len(state) == 0:
          state["step"] = 0
          state["exp_avg"] = torch.zeros_like(grad)
          if factored:
            state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
            state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] +
                                                  grad_shape[-1:]).to(grad)
          else:
            state["exp_avg_sq"] = torch.zeros_like(grad)
        else:
          state["exp_avg"] = state["exp_avg"].to(grad)
          if factored:
            state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
            state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
          else:
            state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

        p_data_fp32 = p.data
        if p.data.dtype in {torch.float16, torch.bfloat16}:
          p_data_fp32 = p_data_fp32.float()

        state["step"] += 1
        lr = group["lr"]
        beta1 = group["beta1"]
        beta2 = group["decay_adam"]

        t = state["step"]
        beta2t = beta2 * (1. - beta2**(t - 1.)) / (1. - beta2**t)

        exp_avg_sq_update = (grad**2) + group["epsilon1_grad_sq_reg"]
        if factored:
          exp_avg_sq_row = state["exp_avg_sq_row"]
          exp_avg_sq_col = state["exp_avg_sq_col"]

          exp_avg_sq_row.mul_(beta2t).add_(
              exp_avg_sq_update.mean(dim=-1), alpha=1.0 - beta2t)
          exp_avg_sq_col.mul_(beta2t).add_(
              exp_avg_sq_update.mean(dim=-2), alpha=1.0 - beta2t)

          r_factor = inf_to_nan(
              exp_avg_sq_row /
              exp_avg_sq_row.mean(dim=-1, keepdim=True)).unsqueeze(-1)
          c_factor = inf_to_nan(exp_avg_sq_col).unsqueeze(-2)
          denom = r_factor * c_factor
        else:
          exp_avg_sq = state["exp_avg_sq"]

          exp_avg_sq.mul_(beta2t).add_(exp_avg_sq_update, alpha=1.0 - beta2t)
          denom = exp_avg_sq

        denom = denom.sqrt()
        update = grad / denom
        # Clip the update based on RMS.
        clipping_denom = inf_to_nan(torch.square(update).mean().sqrt() \
        /group["clip_threshold"]).clamp(min=1.0)
        update = update / clipping_denom * lr
        # Momentum
        exp_avg = state["exp_avg"]
        exp_avg.mul_(beta1).add_(update, alpha=1 - beta1)

        if group["weight_decay"] != 0:
          p_data_fp32.add_(p_data_fp32, alpha=-group["weight_decay"] * lr)

        p_data_fp32.add_(-exp_avg)

        if p.data.dtype in {torch.float16, torch.bfloat16}:
          p.data.copy_(p_data_fp32)

    return loss


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
  optimizer_state['optimizer'].zero_grad()

  logits_batch, new_model_state = workload.model_fn(
      params=current_model,
      augmented_and_preprocessed_input_batch=batch,
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      update_batch_norm=True)

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

  loss.backward()

  with torch.no_grad():
    parameters = [p for p in current_model.parameters() if p.grad is not None]
    grad_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)

  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
    torch.nn.utils.clip_grad_norm_(
        current_model.parameters(), max_norm=grad_clip)
  optimizer_state['optimizer'].step()
  optimizer_state['scheduler'].step()

  # Log training metrics - loss, grad_norm, batch_size.
  if global_step <= 100 or global_step % 500 == 0:
    if workload.metrics_logger is not None:
      workload.metrics_logger.append_scalar_metrics(
          {
              'loss': loss.item(),
              'grad_norm': grad_norm.item(),
          }, global_step)
    logging.info('%d) loss = %0.3f, grad_norm = %0.3f',
                 global_step,
                 loss.item(),
                 grad_norm.item())

  return (optimizer_state, current_param_container, new_model_state)


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262_144
  elif workoad_name == 'criteo1tb_layernorm':
    return 262_144
  elif workload_name == 'criteo1tb_resnet':
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
