"""FastMRI workload implemented in PyTorch."""

import contextlib
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
from skimage.metrics import structural_similarity
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from algorithmic_efficiency import data_utils
from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
import algorithmic_efficiency.random_utils as prng
from algorithmic_efficiency.workloads.fastmri.fastmri_pytorch.input_pipeline import \
    RandomMask
from algorithmic_efficiency.workloads.fastmri.fastmri_pytorch.input_pipeline import \
    SliceDataset
from algorithmic_efficiency.workloads.fastmri.fastmri_pytorch.input_pipeline import \
    UnetDataTransform
from algorithmic_efficiency.workloads.fastmri.fastmri_pytorch.models import \
    unet
from algorithmic_efficiency.workloads.fastmri.workload import \
    BaseFastMRIWorkload

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


def ssim(gt: torch.Tensor,
         pred: torch.Tensor,
         mean: np.ndarray,
         std: np.ndarray,
         volume_max: Optional[float] = None) -> np.ndarray:
  """Compute Structural Similarity Index Metric (SSIM)"""
  # TODO: Change this
  if not len(gt.shape) == 3:
    raise ValueError("Unexpected number of dimensions in ground truth.")
  if not len(gt.shape) == len(pred.shape):
    raise ValueError("Ground truth dimensions does not match pred.")

  if volume_max is None:
    volume_max = torch.ones(gt.shape[0])

  gt = gt * std.view(std.shape[0], 1, 1) + mean.view(mean.shape[0], 1, 1)
  pred = pred * std.view(std.shape[0], 1, 1) + mean.view(mean.shape[0], 1, 1)
  gt = gt.detach().cpu().numpy()
  pred = pred.detach().cpu().numpy()
  volume_max = volume_max.detach().cpu().numpy()

  ssims = 0
  for slice_num in range(gt.shape[0]):
    ssims = ssims + structural_similarity(
        gt[slice_num], pred[slice_num], data_range=volume_max[slice_num])

  return ssims


def worker_init_fn(worker_id):
  """Handle random seeding for all mask_func."""
  worker_info = torch.utils.data.get_worker_info()
  data = worker_info.dataset

  # for NumPy random seed we need it to be in this range
  base_seed = worker_info.seed

  if data.transform.mask_func is not None:
    # DDP training: unique seed is determined by worker and device
    if USE_PYTORCH_DDP:
      seed = base_seed + RANK * worker_info.num_workers
    else:
      seed = base_seed
    data.transform.mask_func.rng.seed(seed % (2**32 - 1))


class FastMRIWorkload(BaseFastMRIWorkload):

  def __init__(self):
    super().__init__()
    self._param_types = None
    self._eval_iters = {}

  @property
  def model_params_types(self):
    """The shapes of the parameters in the workload model."""
    if self._param_types is None:
      self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    return self._param_types

  def build_input_queue(self,
                        data_rng: spec.RandomState,
                        split: str,
                        data_dir: str,
                        global_batch_size: int):
    it = self._build_dataset(data_rng, split, data_dir, global_batch_size)
    for batch in it:
      yield {
          'inputs': batch['inputs'].float().to(DEVICE, non_blocking=True),
          'targets': batch['targets'].to(DEVICE, non_blocking=True),
          'mean': batch['mean'].to(DEVICE, non_blocking=True),
          'std': batch['std'].to(DEVICE, non_blocking=True),
          'fname': batch['fname'],
          'slice_num': batch['slice_num'],
          'volume_max': batch['volume_max']
      }

  def _build_dataset(self,
                     data_rng: spec.RandomState,
                     split: str,
                     data_dir: str,
                     batch_size: int):
    del data_rng
    is_train = split == 'train'
    mask = RandomMask(self.center_fractions, self.accelerations)

    transform_config = {
        'train': UnetDataTransform(mask_func=mask, use_seed=False),
        'eval_train': UnetDataTransform(mask_func=mask),
        'validation': UnetDataTransform(mask_func=mask),
    }

    folder = {'train': 'train', 'validation': 'val', 'eval_train': 'train'}

    dataset = SliceDataset(
        root=os.path.join(data_dir, "singlecoil_" + folder[split]),
        transform=transform_config[split],
    )

    if split == 'eval_train':
      # We always use the same subset of the training data for evaluation.
      dataset = torch.utils.data.Subset(dataset,
                                        range(self.num_eval_train_examples))

    sampler = None
    if USE_PYTORCH_DDP:
      if is_train:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=N_GPUS, rank=RANK, shuffle=True)
      else:
        sampler = data_utils.DistributedEvalSampler(
            dataset, num_replicas=N_GPUS, rank=RANK, shuffle=False)
      batch_size //= N_GPUS
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not USE_PYTORCH_DDP and is_train,
        worker_init_fn=worker_init_fn,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=is_train)

    dataloader = data_utils.cycle(
        dataloader,
        custom_sampler=USE_PYTORCH_DDP,
        keys=('inputs',
              'targets',
              'mean',
              'std',
              'fname',
              'slice_num',
              'volume_max'))

    return dataloader

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model = unet()
    self._param_shapes = {
        k: spec.ShapeTuple(v.shape) for k, v in model.named_parameters()
    }
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = torch.nn.DataParallel(model)
    return model, None

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
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

    if mode == spec.ForwardPassMode.TRAIN:
      model.train()

    contexts = {
        spec.ForwardPassMode.EVAL: torch.no_grad,
        spec.ForwardPassMode.TRAIN: contextlib.nullcontext
    }

    with contexts[mode]():
      targets_batch = model(
          augmented_and_preprocessed_input_batch['inputs'].unsqueeze(
              1)).squeeze(1)

    return targets_batch, None

  def output_activation_fn(self,
                           logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:

    activation_fn = {
        spec.LossType.SOFTMAX_CROSS_ENTROPY: F.softmax,
        spec.LossType.SIGMOID_CROSS_ENTROPY: F.sigmoid,
        spec.LossType.MEAN_SQUARED_ERROR: lambda z: z,
        spec.LossType.MEAN_ABSOLUTE_ERROR: lambda z: z
    }
    return activation_fn[loss_type](logits_batch)

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(self, targets_batch: spec.Tensor,
              outputs_batch: spec.Tensor) -> spec.Tensor:  # differentiable
    return F.l1_loss(
        outputs_batch, targets_batch, reduction='none').mean(dim=(1, 2))

  def _eval_metric(self, outputs, targets, mean, std, volume_max):
    """Return the SSIM and loss as a dict."""
    ssim_vals = ssim(
        targets, outputs, mean=mean, std=std, volume_max=volume_max)
    loss = self.loss_fn(outputs, targets).sum()
    return {'ssim': ssim_vals, 'loss': loss}

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str,
                           global_step: int = 0):
    """Run a full evaluation of the model."""
    data_rng, model_rng = prng.split(rng, 2)
    if split not in self._eval_iters:
      # These iterators repeat indefinitely.
      self._eval_iters[split] = self.build_input_queue(
          data_rng, split, data_dir, global_batch_size=global_batch_size)

    total_metrics = {
        'ssim': torch.tensor(0., device=DEVICE),
        'loss': torch.tensor(0., device=DEVICE),
    }
    num_batches = int(math.ceil(num_examples / global_batch_size))
    for _ in range(num_batches):
      batch = next(self._eval_iters[split])
      outputs, _ = self.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.EVAL,
        model_rng,
        update_batch_norm=False)
      batch_metrics = self._eval_metric(outputs,
                                        batch['targets'],
                                        batch['mean'],
                                        batch['std'],
                                        batch['volume_max'])
      total_metrics = {
          k: v + batch_metrics[k] for k, v in total_metrics.items()
      }
    if USE_PYTORCH_DDP:
      for metric in total_metrics.values():
        dist.all_reduce(metric)
    return {k: float(v.item() / num_examples) for k, v in total_metrics.items()}
