from typing import Dict, Iterable, Optional, Tuple

import jax
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import Sampler

from algorithmic_efficiency import spec


def shard_and_maybe_pad_np(
    batch: Dict[str, spec.Tensor],
    padding_value: int = 0,
    global_batch_size: Optional[int] = None) -> Dict[str, spec.Tensor]:
  """Prepare tf data for JAX or PyTorch DDP.

  Convert an input batch from tf Tensors to numpy arrays, pad it with
  padding_value if the batch size is not divisible by the number of devices,
  create the corresponding mask, and reshape it to be sharded across devices.
  """
  local_device_count = max(torch.cuda.device_count(), jax.local_device_count())
  inputs = batch['inputs']
  current_batch_size = inputs[0].shape[0] if isinstance(
      inputs, tuple) else inputs.shape[0]
  remainder_size = current_batch_size % local_device_count
  if remainder_size != 0:
    if global_batch_size is not None:
      pad_size = global_batch_size - current_batch_size
    else:
      pad_size = local_device_count - remainder_size
    targets = batch['targets']
    targets_shape = tuple(
        targets[0].shape if isinstance(targets, tuple) else targets.shape)
    # We need a 2d mask for WMT.
    mask_shape = targets_shape if len(targets_shape) < 3 else targets_shape[0]
    # The weights will also be padded.
    batch['weights'] = np.ones(mask_shape)

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    if not isinstance(x, np.ndarray):
      x = x._numpy()  # pylint: disable=protected-access

    # Pad if remainder_size != 0 (should only be possible during evaluation).
    if remainder_size != 0:
      x = pad(x, pad_size, 'jax', padding_value=padding_value)

    # Reshape (global_batch_size, ...) to
    # (local_device_count, per_device_batch_size, ...).
    # Assumes that `global_batch_size % local_device_count == 0`.
    return x.reshape((local_device_count, -1, *x.shape[1:]))

  return jax.tree_map(_prepare, batch)


def pad(tensor: spec.Tensor,
        pad_size: int,
        framework: str,
        padding_value: int = 0) -> spec.Tensor:
  if len(tensor) > 1:
    pad_size = (pad_size, *tensor.shape[1:])
  if framework == 'pytorch':
    padding = torch.full(
        pad_size, padding_value, dtype=tensor.dtype, device=tensor.device)
    padded_tensor = torch.cat((tensor, padding), dim=0)
  elif framework == 'jax':
    padding = np.full(pad_size, padding_value, dtype=tensor.dtype)
    padded_tensor = np.concatenate((tensor, padding), axis=0)
  else:
    raise ValueError(f'Framework has to be pytorch or jax, but is {framework}.')
  return padded_tensor


def mixup_pytorch(batch: Tuple[spec.Tensor, spec.Tensor],
                  alpha: float = 0.2) -> Tuple[spec.Tensor, spec.Tensor]:
  inputs, targets = batch
  # Transform to one-hot targets.
  targets = F.one_hot(targets, num_classes=1000)
  # Compute weight for convex combination by sampling from Beta distribution.
  beta_dist = torch.distributions.beta.Beta(alpha, alpha)
  weight = beta_dist.sample()
  # Return convex combination of original and shifted inputs and targets.
  inputs = weight * inputs + (1.0 - weight) * torch.roll(inputs, 1, dims=0)
  targets = weight * targets + (1.0 - weight) * torch.roll(targets, 1, dims=0)
  return (inputs, targets)

# github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py
class DistributedEvalSampler(Sampler):
  r"""
  DistributedEvalSampler is different from DistributedSampler.
  It does NOT add extra samples to make it evenly divisible.
  DistributedEvalSampler should NOT be used for training. The distributed
  processes could hang forever.
  See this issue for details: https://github.com/pytorch/pytorch/issues/22584
  shuffle is disabled by default
  DistributedEvalSampler is for evaluation purpose where synchronization does
  not happen every epoch.
  Synchronization should be done outside the dataloader loop.
  Sampler that restricts data loading to a subset of the dataset.
  It is especially useful in conjunction with
  :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
  process can pass a :class`~DistributedEvalSampler` instance as
  a :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
  original dataset that is exclusive to it.
  .. note::
    Dataset is assumed to be of constant size.
  Arguments:
    dataset: Dataset used for sampling.
    num_replicas (int, optional): Number of processes participating in
        distributed training. By default, :attr:`rank` is retrieved from the
        current distributed group.
    rank (int, optional): Rank of the current process within
        :attr:`num_replicas`. By default, :attr:`rank` is retrieved from the
        current distributed group.
    shuffle (bool, optional): If ``True``, sampler will shuffle the
        indices. Default: ``False``
    seed (int, optional): random seed used to shuffle the sampler if
        :attr:`shuffle=True`. This number should be identical across all
        processes in the distributed group. Default: ``0``.
  .. warning::
    In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>`
    method at the beginning of each epoch **before** creating the
    :class:`DataLoader` iterator is necessary to make shuffling work
    properly across multiple epochs. Otherwise, the same ordering will be
    always used.
  Example::
    >>> sampler = DistributedSampler(dataset) if is_distributed else None
    >>> loader = DataLoader(dataset, shuffle=(sampler is None),
    ...                     sampler=sampler)
    >>> for epoch in range(start_epoch, n_epochs):
    ...     if is_distributed:
    ...         sampler.set_epoch(epoch)
    ...     train(loader)
  """

  def __init__(self,
               dataset,
               num_replicas=None,
               rank=None,
               shuffle=False,
               seed=0):
    if num_replicas is None:
      if not dist.is_available():
        raise RuntimeError('Requires distributed package to be available.')
      num_replicas = dist.get_world_size()
    if rank is None:
      if not dist.is_available():
        raise RuntimeError('Requires distributed package to be available.')
      rank = dist.get_rank()
    self.dataset = dataset
    self.num_replicas = num_replicas
    self.rank = rank
    self.epoch = 0
    # true value without extra samples
    self.total_size = len(self.dataset)
    indices = list(range(self.total_size))
    indices = indices[self.rank:self.total_size:self.num_replicas]
    # true value without extra samples
    self.num_samples = len(indices)

    self.shuffle = shuffle
    self.seed = seed

  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()
    else:
      indices = list(range(len(self.dataset)))

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    return iter(indices)

  def __len__(self):
    return self.num_samples

  def set_epoch(self, epoch):
    r"""
    Sets the epoch for this sampler. When :attr:`shuffle=True`, this
    ensures all replicas use a different random ordering for each epoch.
    Otherwise, the next iteration of this sampler will yield the same
    ordering.
    Arguments:
        epoch (int): _epoch number.
    """
    self.epoch = epoch


# github.com/pytorch/pytorch/issues/23900#issuecomment-518858050
def cycle(iterable: Iterable,
          keys: Tuple[str, ...] = ('inputs', 'targets'),
          custom_sampler: bool = False,
          use_mixup: bool = False,
          mixup_alpha: float = 0.2):
  iterator = iter(iterable)
  epoch = 0
  while True:
    try:
      batch = next(iterator)
      if use_mixup:
        assert keys == ('inputs', 'targets')
        batch = mixup_pytorch(batch, alpha=mixup_alpha)
      assert len(keys) == len(batch)
      yield dict(zip(keys, batch))
    except StopIteration:
      if custom_sampler and isinstance(iterable, DataLoader):
        epoch += 1
        iterable.sampler.set_epoch(epoch)
      iterator = iter(iterable)


# Inspired by
# github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/
# ConvNets/image_classification/dataloaders.py
class PrefetchedWrapper:

  def __init__(self, dataloader, device, start_epoch=0):
    self.dataloader = dataloader
    self.epoch = start_epoch
    self.device = device

  def __len__(self):
    return len(self.dataloader)

  def __iter__(self):
    if isinstance(self.dataloader.sampler, DistributedSampler):
      self.dataloader.sampler.set_epoch(self.epoch)
    self.epoch += 1
    return self.prefetched_loader()

  def prefetched_loader(self):
    stream = torch.cuda.Stream()
    first = True

    for next_inputs, next_targets in self.dataloader:
      with torch.cuda.stream(stream):
        next_inputs = next_inputs.to(
            self.device, dtype=torch.float, non_blocking=True)
        next_targets = next_targets.to(self.device, non_blocking=True)

      if not first:
        yield inputs, targets
      else:
        first = False

      torch.cuda.current_stream().wait_stream(stream)
      inputs = next_inputs
      targets = next_targets

    yield inputs, targets
