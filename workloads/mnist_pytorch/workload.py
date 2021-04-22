"""MNIST workload implemented in Jax."""

import contextlib
import struct
import time
from typing import Tuple

import spec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST

"""
TODO: match network definition in mnist_jax
"""
class _Model(nn.Module):

    def __init__(self, input_size, num_hidden, num_classes=10):
        super(_Model, self).__init__()

        self.net = nn.Sequential(OrderedDict([
            ('layer1',     torch.nn.Linear(input_size, num_hidden, bias=True)),
            ('layer1_sig', torch.nn.Sigmoid()),
            ('layer2',     torch.nn.Linear(num_hidden, num_classes, bias=True)),
            ('output',     torch.nn.LogSoftmax(dim=1))
        ]))

    def forward(self, x: spec.Tensor, train: bool):
        output = self.net(x)

        return output


class MnistWorkload(spec.Workload):

  def __init__(self):
    self._eval_ds = None

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result > 0.9

  def _build_dataloader(self,
      data_rng: spec.RandomState,
      split: str,
      batch_size: int):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])

    train_set = MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = MNIST(DATA_DIR, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               seed=data_rng)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False)
    loaders = {
      'train': train_loader,
      'eval': test_loader
    }

    return loaders[split]


  def build_input_queue(
      self,
      data_rng: spec.RandomState,
      split: str,
      batch_size: int):
    return iter(self._build_dataloader(data_rng, split, batch_size))

  @property
  def param_shapes(self):
    """
    TODO: return shape tuples from model as a tree
    """
    raise NotImplementedError

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def train_mean(self):
    return 0.0

  @property
  def train_stddev(self):
    return 1.0

  def model_params_types(self):
    pass

  @property
  def max_allowed_runtime_sec(self):
    return 60

  @property
  def eval_period_time_sec(self):
    return 10

  # Return whether or not a key in spec.ParameterTree is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def preprocess_for_train(
      self,
      selected_raw_input_batch: spec.Tensor,
      selected_label_batch: spec.Tensor,
      rng: spec.RandomState) -> spec.Tensor:
    del rng
    return self.preprocess_for_eval(
        selected_raw_input_batch, selected_label_batch, None, None)

  def preprocess_for_eval(
      self,
      raw_input_batch: spec.Tensor,
      raw_label_batch: spec.Tensor,
      train_mean: spec.Tensor,
      train_stddev: spec.Tensor) -> spec.Tensor:
    del train_mean
    del train_stddev

    N = raw_input_batch.size()[0]
    raw_input_batch = raw_input_batch.view(N, -1)
    return (raw_input_batch, raw_label_batch)

  _InitState = Tuple[spec.ParameterTree, spec.ModelAuxillaryState]
  def init_model_fn(self, rng: spec.RandomState) -> _InitState:
    torch.random.manual_seed(rng)
    model = _Model(input_size=(28, 28))
    return model, None

  def model_fn(
      self,
      model: spec.ParameterTree,
      augmented_and_preprocessed_input_batch: spec.Tensor,
      model_state: spec.ModelAuxillaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxillaryState]:
    del model_state
    del rng
    del update_batch_norm

    if mode == spec.ForwardPassMode.EVAL:
        model.eval()

        contexts = {
            spec.ForwardPassMode.EVAL: torch.no_grad,
            spec.ForwardPassMode.TRAIN: contextlib.nullcontext
        }

        with contexts[mode]():
            logits_batch = model(augmented_and_preprocessed_input_batch)


    return logits_batch, None

  # LossFn = Callable[Tuple[spec.Tensor, spec.Tensor], spec.Tensor]
  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,
      logits_batch: spec.Tensor,
      loss_type: spec.LossType) -> spec.Tensor:  # differentiable
    del loss_type

    if loss_type is not spec.LossType.SOFTMAX_CROSS_ENTROPY:
      raise NotImplementedError

    # TODO: resolve whether label_batch should be one_hot or not.
    return F.nll_loss(logits_batch, label_batch)

  def _eval_metric(self, logits, labels):
    _, predicted = torch.max(logits.data, 1)
    return (predicted == labels).mean()).cpu().numpy()



  def eval_model(
      self,
      params: spec.ParameterTree,
      model_state: spec.ModelAuxillaryState,
      rng: spec.RandomState):
    """Run a full evaluation of the model."""
    # TODO: use a split rng
    # data_rng, model_rng = jax.random.split(rng, 2)
    data_rng, model_rng = rng[:2]

    eval_batch_size = 2000
    num_batches = 10000 // eval_batch_size
    if self._eval_ds is None:
      self._eval_ds = self._build_dataloader(
          data_rng, split='test', batch_size=eval_batch_size)
    eval_iter = iter(self._eval_ds)
    total_loss = 0.
    total_accuracy = 0.
    for x in eval_iter:
      images = x['image']
      labels = x['label']
      logits, _ = self.model_fn(
          params,
          images,
          model_state,
          spec.ForwardPassMode.EVAL,
          model_rng,
          update_batch_norm=False)
      # TODO(znado): add additional eval metrics?
      # total_loss += self.loss_fn(labels, logits, self.loss_type)
      total_accuracy += self._eval_metric(logits, labels)
    return float(total_accuracy / num_batches)
