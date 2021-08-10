"""Training algorithm track submission functions for LibriSpeech."""
from typing import Iterator, List, Tuple

import spec
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none")


def get_batch_size(workload_name):
  batch_sizes = {"librispeech_pytorch": 8}
  return batch_sizes[workload_name]


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparamters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del workload
  del model_state
  del rng

  optimizer = torch.optim.Adam(model_params.parameters(),
                               hyperparameters.learning_rate)
  return optimizer


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
  del workload
  del current_params_types
  del eval_results
  del global_step
  del model_state
  del loss_type
  del hyperparameters
  del label_batch
  del rng

  current_param_container.train()
  _, features, trns, input_lengths = input_batch
  features = features.float().to(device)
  features = features.transpose(1, 2).unsqueeze(1)
  trns = trns.long().to(device)
  input_lengths = input_lengths.long().to(device)

  optimizer_state.zero_grad()
  log_y, output_lengths = current_param_container(features, input_lengths, trns)

  target_lengths = torch.IntTensor([len(y[y != 0]) for y in trns])
  train_ctc_loss = torch.mean(
      ctc_loss(log_y, trns, output_lengths, target_lengths) /
      (target_lengths.float().to(device)))

  train_ctc_loss.backward()
  optimizer_state.step()

  return optimizer_state, current_param_container, None


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
  del optimizer_state
  del current_param_container
  del global_step
  del rng
  del hyperparameters
  del workload

  return next(input_queue), None

