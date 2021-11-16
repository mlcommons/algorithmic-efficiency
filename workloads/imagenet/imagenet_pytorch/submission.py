"""Training algorithm track submission functions for MNIST."""
import torch
import spec

from workloads.mnist.mnist_pytorch.submission import update_params, data_selection


def get_batch_size(workload_name):
  batch_sizes = {'imagenet_pytorch': 128}
  return batch_sizes[workload_name]


def init_optimizer_state(
    workload: spec.Workload,
    model_params: spec.ParameterContainer,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparamters,
    rng: spec.RandomState) -> spec.OptimizerState:
  del rng
  del model_state
  del workload

  optimizer_state = {
      'optimizer': torch.optim.SGD(model_params.parameters(),
                                   lr=hyperparameters.learning_rate,
                                   momentum=hyperparameters.momentum,
                                   weight_decay=hyperparameters.weight_decay)
  }
  return optimizer_state
