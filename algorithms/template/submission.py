"""Template submission module.

See https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md#allowed-submissions
and https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md#disallowed-submissions
for guidelines.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

from algoperf import spec


def get_batch_size(workload_name: str) -> int:
  """Gets batch size for workload.
  Note that these batch sizes only apply during training and not during evals.

  Args:
      workload_name (str): Valid workload_name values are: "wmt", "ogbg",
      "criteo1tb", "fastmri", "imagenet_resnet", "imagenet_vit",
      "librispeech_deepspeech", "librispeech_conformer" or any of the
      variants.

  Returns:
      int: batch_size

  Raises:
      ValueError: If workload_name is not handled.
  """
  pass


def init_optimizer_state(
  workload: spec.Workload,
  model_params: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  rng: spec.RandomState,
) -> spec.OptimizerState:
  """Initializes optimizer state, e.g. for EMAs or learning rate schedules.

  Args:
    workload (spec.Workload): The current workload.
    model_params (spec.ParameterContainer): The current model parameters.
    model_state (spec.ModelAuxiliaryState): Holds auxiliary state of the model,
    such as current batch norm statistics.
    hyperparameters (spec.Hyperparameters): The hyperparameters for the
    algorithm.
    rng (spec.RandomState): The random state.

  Returns:
    spec.OptimizerState: The initialized optimizer state.
  """
  pass


def update_params(
  workload: spec.Workload,
  current_param_container: spec.ParameterContainer,
  current_params_types: spec.ParameterTypeTree,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  batch: Dict[str, spec.Tensor],
  loss_type: spec.LossType,
  optimizer_state: spec.OptimizerState,
  eval_results: List[Tuple[int, float]],
  global_step: int,
  rng: spec.RandomState,
  train_state: Optional[Dict[str, Any]] = None,
) -> spec.UpdateReturn:
  """Updates the model parameters, e.g., a step of the training algorithm.

  Args:
      workload (spec.Workload): The current workload.
      current_param_container (spec.ParameterContainer): The current model
      parameters.
      current_params_types (spec.ParameterTypeTree): The types of the current
      model parameters, e.g. weights, biases, conv, batch norm, etc.
      model_state (spec.ModelAuxiliaryState): Holds auxiliary state of the
      model, such as current batch norm statistics.
      hyperparameters (spec.Hyperparameters): The hyperparameters for the
      algorithm.
      batch (Dict[str, spec.Tensor]): The current batch of data.
      loss_type (spec.LossType): The type of loss function.
      optimizer_state (spec.OptimizerState): The current optimizer state.
      eval_results (List[Tuple[int, float]]): The evaluation results from the
      previous step.
      global_step (int): The current global step.
      rng (spec.RandomState): The random state.
      train_state (Optional[Dict[str, Any]], optional): The current training
      state, e.g., accumulated submission time.

  Returns:
      spec.UpdateReturn: Tuple[OptimizerState, ParameterContainer,
      ModelAuxiliaryState], containing the new optimizer state, the new
      parameters, and the new model state.
  """
  pass


def prepare_for_eval(
  workload: spec.Workload,
  current_param_container: spec.ParameterContainer,
  current_params_types: spec.ParameterTypeTree,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  loss_type: spec.LossType,
  optimizer_state: spec.OptimizerState,
  eval_results: List[Tuple[int, float]],
  global_step: int,
  rng: spec.RandomState,
) -> spec.UpdateReturn:
  """Prepares for evaluation, e.g., switching to evaluation parameters.
  Arguments are the same as `update_params`, with the only exception of batch.

  Args:
      workload (spec.Workload): The current workload.
      current_param_container (spec.ParameterContainer): The current model
      parameters.
      current_params_types (spec.ParameterTypeTree): The types of the current
      model parameters, e.g. weights, biases, conv, batch norm, etc.
      model_state (spec.ModelAuxiliaryState): Holds auxiliary state of the
      model, such as current batch norm statistics.
      hyperparameters (spec.Hyperparameters): The hyperparameters for the
      algorithm.
      loss_type (spec.LossType): The type of loss function.
      optimizer_state (spec.OptimizerState): The current optimizer state.
      eval_results (List[Tuple[int, float]]): The evaluation results from the
      previous step.
      global_step (int): The current global step.
      rng (spec.RandomState): The random state.

  Returns:
      spec.UpdateReturn: Tuple[OptimizerState, ParameterContainer,
      ModelAuxiliaryState], containing the new optimizer state, the new
      parameters, and the new model state.
  """
  pass


def data_selection(
  workload: spec.Workload,
  input_queue: Iterator[Dict[str, spec.Tensor]],
  optimizer_state: spec.OptimizerState,
  current_param_container: spec.ParameterContainer,
  model_state: spec.ModelAuxiliaryState,
  hyperparameters: spec.Hyperparameters,
  global_step: int,
  rng: spec.RandomState,
) -> Dict[str, spec.Tensor]:
  """Selects data from the infinitely repeating, pre-shuffled input queue.
  Each element of the queue is a batch of training examples and labels.
  Tip: If you would just like the next batch from the input queue return
  `next(input_queue)`.

  Args:
      workload (spec.Workload): The current workload.
      input_queue (Iterator[Dict[str, spec.Tensor]]): The input queue.
      optimizer_state (spec.OptimizerState): The current optimizer state.
      current_param_container (spec.ParameterContainer): The current model
      model_state (spec.ModelAuxiliaryState): Holds auxiliary state of the
      model, such as current batch norm statistics.
      hyperparameters (spec.Hyperparameters): The hyperparameters for the
      algorithm.
      global_step (int): The current global step.
      rng (spec.RandomState): The random state.

  Returns:
      Dict[str, spec.Tensor]: A dictionary of tensors, where the keys are the
      names of the input features and the values are the corresponding tensors.
  """
  pass
