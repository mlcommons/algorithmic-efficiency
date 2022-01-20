"""ImageNet workload implemented in PyTorch."""
import contextlib
import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder

import spec
import random_utils as prng
from workloads.imagenet.workload import ImagenetWorkload
from workloads.imagenet.imagenet_pytorch.models import resnet50
from workloads.imagenet.imagenet_pytorch.utils import cycle


DEVICE='cuda' if torch.cuda.is_available() else 'cpu'


class ImagenetWorkload(ImagenetWorkload):

  def __init__(self):
    super().__init__()
    self.dataset = 'imagenet2012'

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
    return 1281167

  @property
  def num_eval_examples(self):
    return 50000

  @property
  def param_shapes(self):
    """
    TODO: return shape tuples from model as a tree
    """
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
    n_data = 0
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
      n_data += batch_metrics['n_data']
    return {k: float(v / n_data) for k, v in total_metrics.items()}

  def _build_dataset(self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      batch_size: int):

    is_train = (split == "train")

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[i/255 for i in self.train_mean], 
            std=[i/255 for i in self.train_stddev])
    ])
    transform_config = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(
                self.center_crop_size,
                scale=self.scale_ratio_range,
                ratio=self.aspect_ratio_range),
            transforms.RandomHorizontalFlip(),
            normalize]),
        "test": transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(self.center_crop_size),
            normalize])
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
        # Note (runame): not sure why we do this?
        drop_last=is_train
    )

    if is_train:
      dataloader = cycle(dataloader)

    return dataloader

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

  def _update_batch_norm(self, model, update_batch_norm):
    for m in model.modules():
      if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        if not update_batch_norm:
          m.eval()
        m.requires_grad_(update_batch_norm)
        m.track_running_stats = update_batch_norm

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

    model = params

    if mode == spec.ForwardPassMode.EVAL:
      if update_batch_norm:
        raise ValueError('Batch norm statistics cannot be updated during evaluation.') 
      model.eval()

    if mode == spec.ForwardPassMode.TRAIN:
      model.train()
      self._update_batch_norm(model, update_batch_norm)

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

    return F.cross_entropy(logits_batch, label_batch, reduction='sum')

  def _eval_metric(self, logits, labels):
    """Return the mean accuracy and loss as a dict."""
    predicted = torch.argmax(logits, 1)
    accuracy = (predicted == labels).cpu().numpy().sum()  # not accuracy, but nr. of correct predictions
    loss = self.loss_fn(labels, logits).item()
    n_data = len(logits)
    return {'accuracy': accuracy, 'loss': loss, 'n_data': n_data}
