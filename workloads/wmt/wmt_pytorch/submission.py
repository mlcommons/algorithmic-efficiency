from typing import Iterator, List, Tuple, Union

from . import workload
from transformer.Optim import ScheduledOptim
import torch
import spec
from train import train_epoch
DEVICE = 'cuda'


def get_batch_size(workload_name):
    batch_sizes = {'wmt_pytorch': 16}
    return batch_sizes[workload_name]


def init_optimizer_state(
        workload: spec.Workload,
        model_params: spec.ParameterContainer,
        model_state: spec.ModelAuxiliaryState,
        hyperparameters: spec.Hyperparamters,
        rng: spec.RandomState
) -> spec.OptimizerState:
    del rng
    del model_state
    del workload

    optimizer_state = {
        'optimizer': torch.optim.Adam(model_params.parameters(),
            lr=hyperparameters.learning_rate,)
    }

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

    del current_param_container
    del current_params_types
    del eval_results
    del global_step

    del loss_type
    del hyperparameters
    del label_batch


    current_model = current_param_container
    current_model.train()
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

    return (optimizer_state, current_param_container, new_model_state)