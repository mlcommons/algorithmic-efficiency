import itertools
import os
from typing import Callable, OrderedDict, Tuple

import numpy as np
import torch
from specification.draft_api import *
from torchvision import transforms
from torchvision.datasets import MNIST

from models.base import ReferenceModel

NUM_HIDDEN = 128
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
DATA_DIR = os.path.expanduser('~/')

def sigmoid(z: Tensor) -> Tensor:
    s = 1. / (1. + np.exp(-z))
    return s

def softmax(z: Tensor) -> Tensor:
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def one_hot(Y, num_classes=10):
    Y = Y.reshape(1, -1)
    Y_new = np.eye(num_classes)[Y.astype('int')]
    Y_new = Y_new.T.reshape(num_classes, -1)

    return Y_new

# use pytorch dataloaders for convenience
def _get_dataloaders(data_dir: str,
                     batch_size: int):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_set = MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = MNIST(data_dir, train=False, transform=transform)

    loader_kwargs = {'batch_size': batch_size,
                    'shuffle': True,
                    }

    train_loader = torch.utils.data.DataLoader(train_set, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **loader_kwargs)

    return (iter(train_loader), test_loader)


class MNISTNumpy(ReferenceModel):

    """
    Fixed functions
    """
    @staticmethod
    def preprocess_for_train(selected_raw_input_batch: Tensor,
                             selected_label_batch: Tensor,
                             train_mean: Tensor,
                             train_stddev: Tensor,
                             seed: Seed) -> Tuple[Tensor, Tensor]:

        X = (selected_raw_input_batch - train_mean) / train_stddev
        Y = selected_label_batch
        return (X.numpy().T, Y.numpy())

    @staticmethod
    def preprocess_for_eval(raw_input_batch: Tensor,
                            train_mean: Tensor,
                            train_stddev: Tensor) -> Tensor:

        X = (raw_input_batch - train_mean) / train_stddev
        return X.numpy().T

    @staticmethod
    def init_model_fn(param_shapes: ParameterShapeTree,
                      seed: Seed) -> ParameterTree:
        np.random.seed(seed)

        params = OrderedDict()

        # MNIST is linear, no nested tree
        for name, shape in param_shapes.items():
            fan_out = shape[0]

            k = np.sqrt(1. / fan_out)
            params[name] = np.random.uniform(-1 * k, k, size=shape)

        return params

    @staticmethod
    def model_fn(params: ParameterTree,
                 preprocessed_input_batch: Tensor,
                 mode: ForwardPassMode,
                 seed: Seed,
                 update_batch_norm: bool) -> Tuple[OutputTree, Tensor]:

        # First linear layer
        Z1 = np.matmul(params['W1'], preprocessed_input_batch) + params['b1']
        A1 = sigmoid(Z1)

        # second linear layer
        Z2 = np.matmul(params['W2'], A1) + params['b2']

        # softmax
        A2 = softmax(Z2)

        outputs = OrderedDict([
            ('Z1', Z1),
            ('A1', A1),
            ('Z2', Z2),
            ('A2', A2)
        ])
        return outputs, outputs['A2']

    @staticmethod
    def loss_fn(label_batch: Tensor,
                logits_batch: Tensor,
                loss_type: LossType) -> float:

        if loss_type is not LossType.SOFTMAX_CROSS_ENTROPY:
            raise NotImplementedError

        L_sum = np.sum(np.multiply(label_batch, np.log(logits_batch)))
        m = label_batch.shape[1]
        L = -(1. / m) * L_sum

        return L

    @staticmethod
    def output_activation_fn(logits_batch: Tensor,
                             loss_type: LossType) -> Tensor:
        activation_fn = {
            LossType.SOFTMAX_CROSS_ENTROPY: softmax,
            LossType.SIGMOID_CROSS_ENTROPY: sigmoid,
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
        return MNISTNumpy.model_fn

    @staticmethod
    def eval_metric(logits_batch: Tensor,
                    label_batch: Tensor) -> float:
        label_batch = label_batch.numpy()

        prediction = np.argmax(logits_batch, axis=0)
        p_correct = np.sum(label_batch == prediction) / len(prediction)

        return p_correct


class MNISTNumpySubmission:
    """
    Submission functions
    """
    @staticmethod
    def init_optimizer_state(params_shapes: ParameterShapeTree,
                             hyperparameters: Hyperparameters,
                             seed: Seed) -> OptimizerState:

        optimizer_state = OrderedDict()

        for name, shape in params_shapes.items():
            optimizer_state[name] = np.zeros(shape)

        return optimizer_state

    @staticmethod
    def update_params(current_params: ParameterTree,
                      current_params_types: ParameterTypeTree,
                      hyperparameters: Hyperparameters,
                      preprocessed_input_batch: Tensor,
                      label_batch: Tensor,
                      loss_type: LossType,
                      model_fn: Callable,
                      optimizer_state: OptimizerState,
                      global_step: int,
                      seed: Seed) -> Tuple[OptimizerState, ParameterTree]:

        Y_onehot = one_hot(label_batch)

        # forward pass
        output_batch, logits = model_fn(params=current_params,
                                   preprocessed_input_batch=preprocessed_input_batch,
                                   mode=ForwardPassMode.TRAIN,
                                   seed=seed,
                                   update_batch_norm=False)

        # get gradients
        grads = MNISTNumpySubmission._backward(output=output_batch,
                                     X=preprocessed_input_batch,
                                     Y=Y_onehot,
                                     params=current_params)

        # apply momentum
        updated_gradients = OrderedDict()
        for (name, grad), (new_name, new_grads) in zip(optimizer_state.items(), grads.items()):
            assert name == new_name, "Expecting same order of keys."

            g = (hyperparameters['beta'] * grad + (1. - hyperparameters['beta']) * new_grads)
            updated_gradients[name] = g

        # gradient descent
        updated_params = OrderedDict()
        for (name, variable), (name_g, grad) in zip(current_params.items(), updated_gradients.items()):
            assert name == name_g, "Expecting some order of keys."

            v = variable - hyperparameters['lr'] * grad
            updated_params[name] = v

        return (updated_gradients, updated_params)

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

        # error
        dZ2 = output['A2'] - Y

        # gradients at last layer
        dW2 = np.matmul(dZ2, output['A1'].T)
        db2 = np.sum(dZ2, axis=1, keepdims=True)

        # backward through first layer
        dA1 = np.matmul(params['W2'].T, dZ2)
        dZ1 = dA1 * sigmoid(output['Z1']) * (1 - sigmoid(output['Z1']))

        # gradients at first layer
        dW1 = np.matmul(dZ1, X.T)
        db1 = np.sum(dZ1, axis=1, keepdims=True)

        grads = OrderedDict([
            ('W1', dW1),
            ('b1', db1),
            ('W2', dW2),
            ('b2', db2)
        ])

        return grads


class MNISTWorkload:

    """
    Workload specific data. These are framework-independant
    """
    param_shapes = OrderedDict([
        ('W1', (NUM_HIDDEN, INPUT_SIZE)),
        ('b1', (NUM_HIDDEN, 1)),
        ('W2', (NUM_CLASSES, NUM_HIDDEN)),
        ('b2', (NUM_CLASSES, 1))
    ])

    model_params_types = OrderedDict([
        ('W1', ParamType.WEIGHT),
        ('b1', ParamType.BIAS),
        ('W2', ParamType.WEIGHT),
        ('b2', ParamType.BIAS)
    ])

    train_mean = 0.1307
    train_stddev = 0.3081
    loss_type = LossType.SOFTMAX_CROSS_ENTROPY
    max_allowed_runtime = 600
    eval_period_time = 10
    target_metric_value = 0.97
    comparison_direction = ComparisonDirection.MAXIMIZE

if __name__ == '__main__':
    hyperparameters = {
        "batch_size": 64,
        "lr": 0.01,
        "beta": 0.9,
        "seed": 123
    }

    submission_time, global_step = train_once(
        workload=MNISTWorkload,
        reference=MNISTNumpy,
        init_optimizer_state=MNISTNumpySubmission.init_optimizer_state,
        update_params=MNISTNumpySubmission.update_params,
        data_selection=MNISTNumpySubmission.data_selection,
        hyperparameters=hyperparameters,
        seed=hyperparameters['seed']
    )

    print(f'Target quality met at step {global_step} and time: {submission_time:.2f}.')
