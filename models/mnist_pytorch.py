import contextlib
import itertools
import os
from typing import Callable, OrderedDict, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from specification.draft_api import (ComparisonDirection, ForwardPassMode,
                                     Hyperparameters, InputQueue,
                                     InputQueueMode, LossType, OptimizerState, OutputTree,
                                     ParameterKey, ParameterShapeTree,
                                     ParameterTree, ParameterTypeTree,
                                     ParamType, Seed, Shape, Steps, Tensor,
                                     Timing, train_once)
from torchvision import transforms
from torchvision.datasets import MNIST
from models.base import ReferenceModel

NUM_HIDDEN = 128
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
DATA_DIR = os.path.expanduser('~/')
DEVICE = 'cuda'

Model = torch.nn.Module

# use pytorch dataloaders to convenience
def _get_dataloaders(data_dir: str,
                     batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])

    train_set = MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = MNIST(data_dir, train=False, transform=transform)

    loader_kwargs = {'batch_size': batch_size,
                    'shuffle': True,
                    }

    train_loader = torch.utils.data.DataLoader(train_set, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **loader_kwargs)

    return (iter(train_loader), test_loader)


class Net(nn.Module):
    def __init__(self, num_hidden, input_size, num_classes):
        super(Net, self).__init__()

        self.net = nn.Sequential(OrderedDict([
            ('layer1',     torch.nn.Linear(input_size, num_hidden, bias=True)),
            ('layer1_sig', torch.nn.Sigmoid()),
            ('layer2',     torch.nn.Linear(num_hidden, num_classes, bias=True)),
            ('output',     torch.nn.LogSoftmax(dim=1))
        ]))

    def forward(self, x: Tensor) -> Tuple[OutputTree, Tensor]:
        # output = self.net(x)
        outputs = OrderedDict()

        for name, layer in self.net.named_children():
            x = layer(x)
            outputs[name] = x

        return outputs, outputs['output']
"""
Cheats
"""


class MNISTPyTorch(ReferenceModel):

    def __init__(self, num_hidden=NUM_HIDDEN,
                       input_size=INPUT_SIZE,
                       num_classes=NUM_CLASSES):
        """
        Cheats
        """
        self.MODEL = Net(num_hidden, input_size, num_classes)

    """
    Fixed functions
    """
    @staticmethod
    def preprocess_for_train(selected_raw_input_batch: Tensor,
                             selected_label_batch: Tensor,
                             train_mean: Tensor,
                             train_stddev: Tensor,
                             seed: Seed) -> Tuple[Tensor, Tensor]:

        N, C, H, W = selected_raw_input_batch.size()
        input_batch = selected_raw_input_batch.view(N, -1).to(DEVICE)
        selected_label_batch = selected_label_batch.to(DEVICE)
        return (input_batch, selected_label_batch)

    @staticmethod
    def preprocess_for_eval(raw_input_batch: Tensor,
                            train_mean: Tensor,
                            train_stddev: Tensor) -> Tensor:

        N, C, H, W = raw_input_batch.size()
        input_batch = raw_input_batch.view(N, -1).to(DEVICE)

        return input_batch

    @staticmethod
    def _init_model_fn(param_shapes: ParameterShapeTree,
                       seed: Seed) -> ParameterTree:

        torch.random.manual_seed(seed)
        params = OrderedDict()
        """
        Memory inefficient. Duplicates the memory usage,
        once for creation of params, and again for
        the layer.weights. Also does not use
        pytorch layer's in-place reset_parameters()
        """
        for name, shape in param_shapes.items():
            params[name] = torch.empty(shape)
            torch.nn.init.kaiming_uniform(params[name])

        return params

    """
    * Changed to return PyTorch Model
    * param_shapes is now the model architecture, not the
      tensor shapes. Potentially can have a utility
      function to translate from param_shapes -> condensed arch
    """
    @staticmethod
    def init_model_fn(param_shapes: ParameterShapeTree,
                      seed: Seed) -> Model:
        torch.random.manual_seed(seed)

        model = Net(num_hidden=param_shapes['num_hidden'],
                    input_size=param_shapes['input_size'],
                    num_classes=param_shapes['num_classes'])
        model.net.to(DEVICE)
        return model


    """
    * Changes to accept torch.nn.Module instead of ParameterTree.

    Conforming to existing API is too slow. Other option would be to,
    at every function call:
    (1) reinitialize model
    (2) load model.state_dict()
    """
    @staticmethod
    def model_fn(model: Model,
                 preprocessed_input_batch: Tensor,
                 mode: ForwardPassMode,
                 seed: Seed,
                 update_batch_norm: bool) -> Tuple[OutputTree, Tensor]:

        if mode == ForwardPassMode.EVAL:
            model.eval()

        contexts = {
            ForwardPassMode.EVAL: torch.no_grad,
            ForwardPassMode.TRAIN: contextlib.nullcontext
        }

        with contexts[mode]():
            output_batch, logits = model(preprocessed_input_batch)

        return output_batch, logits

    @staticmethod
    def loss_fn(label_batch: Tensor,
                logits_batch: Tensor,
                loss_type: LossType) -> float:

        if loss_type is not LossType.SOFTMAX_CROSS_ENTROPY:
            raise NotImplementedError

        return F.nll_loss(logits_batch, label_batch)

    @staticmethod
    def output_activation_fn(logits_batch: Tensor,
                             loss_type: LossType) -> Tensor:
        activation_fn = {
            LossType.SOFTMAX_CROSS_ENTROPY: F.softmax,
            LossType.SIGMOID_CROSS_ENTROPY: F.sigmoid,
            LossType.MEAN_SQUARED_ERROR: lambda z: z
        }

        return activation_fn[loss_type](logits_batch)


    @staticmethod
    def _build_input_queue(mode: InputQueueMode,
                           batch_size: int,
                           seed: Seed) -> InputQueue:
        (train_loader, test_loader) = _get_dataloaders(data_dir=DATA_DIR,
                                                       batch_size=batch_size)

        loaders = {
            InputQueueMode.TRAIN: itertools.cycle(train_loader),
            InputQueueMode.EVAL: test_loader
        }

        return loaders[mode]

    @staticmethod
    # unclear purpose of this function?
    def _build_model_fn():
        return MNISTPyTorch.model_fn

    @staticmethod
    def eval_metric(logits_batch: Tensor,
                    label_batch: Tensor) -> float:
        logits_batch = logits_batch.to(DEVICE)
        label_batch = label_batch.to(DEVICE)

        _, predicted = torch.max(logits_batch.data, 1)
        correct = (predicted == label_batch).sum()
        accuracy = correct / len(label_batch)
        return accuracy.cpu().numpy()


class MNISTPyTorchSubmission:
    """
    Submission functions
    """

    """
    * Changed to additionally pass in model.

    If changes to `draft_api.py`, can be further restricted to just
    be Iterable[Tensor].
    """
    @staticmethod
    def init_optimizer_state(param_shapes: ParameterShapeTree,
                             hyperparameters: Hyperparameters,
                             model: Model,
                             seed: Seed) -> OptimizerState:

        optimizer_state = {
            'optimizer': torch.optim.Adam(model.parameters(),
                                          lr=hyperparameters['lr'])
        }
        return optimizer_state

    """
    * Changed to accept the PyTorch Model, instead of parameters

    """
    @staticmethod
    def update_params(current_model: Model,
                      current_params_types: ParameterTypeTree,
                      hyperparameters: Hyperparameters,
                      preprocessed_input_batch: Tensor,
                      label_batch: Tensor,
                      loss_type: LossType,
                      model_fn: Callable,
                      optimizer_state: OptimizerState,
                      global_step: int,
                      seed: Seed) -> Tuple[OptimizerState, Model]:

        current_model.train()

        preprocessed_input_batch = preprocessed_input_batch.to(DEVICE)
        label_batch = label_batch.to(DEVICE)
        optimizer_state['optimizer'].zero_grad()

        _, output = model_fn(model=current_model,
                             preprocessed_input_batch=preprocessed_input_batch,
                             mode=ForwardPassMode.TRAIN,
                             seed=seed,
                             update_batch_norm=False)
        # needs loss _n?
        loss = F.nll_loss(output, label_batch)

        loss.backward()
        optimizer_state['optimizer'].step()

        return (optimizer_state, current_model)

    @staticmethod
    def data_selection(input_queue: InputQueue,
                       optimizer_state: OptimizerState,
                       current_params: ParameterTree,
                       loss_type: LossType,
                       hyperparameters: Hyperparameters,
                       global_step: int,
                       seed: Seed) -> Tuple[Tensor, Tensor]:
        return next(input_queue)

    """
    backward function
    """
    @staticmethod
    def _backward(output, X, Y, params):

        return None


class MNISTWorkload:

    """
    Workload specific data. These are framework-independant
    """
    # param_shapes = OrderedDict([
    #     ('W1', (NUM_HIDDEN, INPUT_SIZE)),
    #     ('b1', (NUM_HIDDEN, 1)),
    #     ('W2', (NUM_CLASSES, NUM_HIDDEN)),
    #     ('b2', (NUM_CLASSES, 1))
    # ])
    param_shapes = {'num_hidden': NUM_HIDDEN,
                    'num_classes': NUM_CLASSES,
                    'input_size': INPUT_SIZE}

    model_params_types = OrderedDict([
        ('W1', ParamType.WEIGHT),
        ('b1', ParamType.BIAS),
        ('W2', ParamType.WEIGHT),
        ('b2', ParamType.BIAS)
    ])

    train_mean = (0.1307, )
    train_stddev = (0.3081, )
    loss_type = LossType.SOFTMAX_CROSS_ENTROPY
    max_allowed_runtime = 600
    eval_period_time = 10
    target_metric_value = 0.975
    comparison_direction = ComparisonDirection.MAXIMIZE

if __name__ == '__main__':
    hyperparameters = {
        "batch_size": 64,
        "lr": 0.001,
        "seed": 123
    }

    train_once(
        workload=MNISTWorkload,
        reference=MNISTPyTorch,
        init_optimizer_state=MNISTPyTorchSubmission.init_optimizer_state,
        update_params=MNISTPyTorchSubmission.update_params,
        data_selection=MNISTPyTorchSubmission.data_selection,
        hyperparameters=hyperparameters,
        seed=hyperparameters['seed']
    )
