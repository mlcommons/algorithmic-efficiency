"""Resnet workload implemented in PyTorch."""
import contextlib
import os
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageNet
from torchvision.datasets.folder import ImageFolder

import spec
import random_utils as prng
from workloads.imagenet.imagenet_pytorch.resnet import resnet50
from workloads.imagenet.imagenet_pytorch.utils import fast_collate, cycle


DEVICE='cuda' if torch.cuda.is_available() else 'cpu'


class ImagenetWorkload(spec.Workload):

  def __init__(self):
    self._eval_ds = None

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['accuracy'] > self.target_value

  @property
  def target_value(self):
    return 0.76

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def train_mean(self):
    return (0.485, 0.456, 0.406)

  @property
  def train_stddev(self):
    return (0.229, 0.224, 0.225)

  """ data augmentation settings """
  @property
  def scale_ratio_range(self):
    return (0.08, 1.0)

  @property
  def aspect_ratio_range(self):
    return (0.75, 4.0 / 3.0)

  @property
  def center_crop_size(self):
    return 224

  @property
  def resize_size(self):
    return 256

  @property
  def num_train_examples(self):
    return 1271167

  @property
  def num_eval_examples(self):
    return 100000

  @property
  def max_allowed_runtime_sec(self):
    return 111600  # 31 hours

  @property
  def eval_period_time_sec(self):
    return 6000  # 100 mins

  def _eval_metric(self, logits, labels):
    """Return the mean accuracy and loss as a dict."""
    raise NotImplementedError

  def eval_model(
      self,
      params: spec.ParameterContainer,
      model_state: spec.ModelAuxiliaryState,
      rng: spec.RandomState,
      data_dir: str):
    """Run a full evaluation of the model."""
    data_rng, model_rng = prng.split(rng, 2)
    eval_batch_size = 128
    if self._eval_ds is None:
      self._eval_ds = self._build_dataset(
          data_rng, 'test', data_dir, batch_size=eval_batch_size)

    total_metrics = {
        'accuracy': 0.,
        'loss': 0.,
    }
    num_batches = 0
    for (images, labels) in self._eval_ds:
      images, labels = self.preprocess_for_eval(images, labels, None, None)
      logits, _ = self.model_fn(
          params,
          images,
          model_state,
          spec.ForwardPassMode.EVAL,
          model_rng,
          update_batch_norm=False)
      # TODO(znado): add additional eval metrics?
      batch_metrics = self._eval_metric(logits, labels)
      total_metrics = {
          k: v + batch_metrics[k] for k, v in total_metrics.items()
      }
      num_batches += 1

    return {k: float(v / num_batches) for k, v in total_metrics.items()}

  def _build_dataset(self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      batch_size: int):

    is_train = (split == "train")

    transform_config = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(
                self.center_crop_size,
                scale=self.scale_ratio_range,
                ratio=self.aspect_ratio_range),
            transforms.RandomHorizontalFlip()]),
        "test": transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(self.center_crop_size)])
    }

    folder = {'train': 'train', 'test': 'val'}

    dataset = ImageFolder(
        os.path.join(data_dir, folder[split]),
        transform=transform_config[split])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=5,
        pin_memory=True,
        drop_last=is_train,
        collate_fn=fast_collate
    )

    if is_train:
      dataloader = cycle(dataloader)

    return dataloader

  def build_input_queue(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      batch_size: int):
    return iter(self._build_dataset(data_rng, split, data_dir, batch_size))

  @property
  def param_shapes(self):
    """
    TODO: return shape tuples from model as a tree
    """
    raise NotImplementedError

  def model_params_types(self):
    pass

  # Return whether or not a key in spec.ParameterTree is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    raise NotImplementedError

  def preprocess_for_train(
      self,
      selected_raw_input_batch: spec.Tensor,
      selected_label_batch: spec.Tensor,
      train_mean: spec.Tensor,
      train_stddev: spec.Tensor,
      rng: spec.RandomState) -> spec.Tensor:
    del train_mean
    del train_stddev
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
    return (raw_input_batch.float().to(DEVICE), raw_label_batch.to(DEVICE))

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model = resnet50()
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    model.to(DEVICE)
    return model, None

  def model_fn(
      self,
      params: spec.ParameterContainer,
      input_batch: spec.Tensor,
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del rng
    del update_batch_norm

    model = params

    if mode == spec.ForwardPassMode.EVAL:
        model.eval()

    contexts = {
      spec.ForwardPassMode.EVAL: torch.no_grad,
      spec.ForwardPassMode.TRAIN: contextlib.nullcontext
    }

    with contexts[mode]():
      logits_batch = model(input_batch)

    return logits_batch, None

  def output_activation_fn(
      self,
      logits_batch: spec.Tensor,
      loss_type: spec.LossType) -> spec.Tensor:

    activation_fn = {
      spec.LossType.SOFTMAX_CROSS_ENTROPY: F.softmax,
      spec.LossType.SIGMOID_CROSS_ENTROPY: F.sigmoid,
      spec.LossType.MEAN_SQUARED_ERROR: lambda z: z
    }

    return activation_fn[loss_type](logits_batch)

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,
      logits_batch: spec.Tensor) -> spec.Tensor:  # differentiable

    return F.nll_loss(logits_batch, label_batch)

  def _eval_metric(self, logits, labels):
    """Return the mean accuracy and loss as a dict."""
    _, predicted = torch.max(logits.data, 1)
    accuracy = (predicted == labels).cpu().numpy().mean()
    loss = self.loss_fn(labels, logits).cpu().numpy().mean()
    return {'accuracy': accuracy, 'loss': loss}
