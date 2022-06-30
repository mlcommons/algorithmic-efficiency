import jax
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import Sampler


def shard_numpy_ds(xs):
  """Prepare tf data for JAX or PyTorch DDP.

  Convert an input batch from tf Tensors to numpy arrays and reshape it to be
  sharded across devices.
  """
  local_device_count = jax.local_device_count()

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # Reshape (global_batch_size, ...) to
    # (local_device_count, per_device_batch_size, ...).
    # Assumes that `global_batch_size % local_device_count == 0`.
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


# github.com/pytorch/pytorch/issues/23900#issuecomment-518858050
def cycle(iterable, keys=('inputs', 'targets'), custom_sampler=False):
  iterator = iter(iterable)
  epoch = 0
  while True:
    try:
      batch = next(iterator)
      assert len(keys) == len(batch)
      yield dict(zip(keys, batch))
    except StopIteration:
      if custom_sampler and isinstance(iterable, DataLoader):
        epoch += 1
        iterable.sampler.set_epoch(epoch)
      iterator = iter(iterable)


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


# github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/
# ConvNets/image_classification/dataloaders.py
def fast_collate(batch, memory_format=torch.contiguous_format):
  imgs = [img[0] for img in batch]
  targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
  w = imgs[0].size[0]
  h = imgs[0].size[1]
  tensor = torch.zeros(
      (len(imgs), 3, h, w),
      dtype=torch.uint8).contiguous(memory_format=memory_format)
  for i, img in enumerate(imgs):
    nump_array = np.asarray(img, dtype=np.uint8)
    if nump_array.ndim < 3:
      nump_array = np.expand_dims(nump_array, axis=-1)
    nump_array = np.rollaxis(nump_array, 2)
    tensor[i] += torch.from_numpy(nump_array.copy())
  return tensor, targets


# Inspired by
# github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/
# ConvNets/image_classification/dataloaders.py
class PrefetchedWrapper:

  def __init__(self, dataloader, device, mean, std, start_epoch=0):
    self.dataloader = dataloader
    self.epoch = start_epoch
    self.device = device
    self.data_mean = torch.tensor([i / 255 for i in mean],
                                  device=device).view(1, 3, 1, 1)
    self.data_std = torch.tensor([i / 255 for i in std],
                                 device=device).view(1, 3, 1, 1)

  def __len__(self):
    return len(self.dataloader)

  def __iter__(self):
    if isinstance(self.dataloader.sampler,
                  (DistributedSampler, DistributedEvalSampler)):
      self.dataloader.sampler.set_epoch(self.epoch)
    self.epoch += 1
    return self.prefetched_loader()

  def prefetched_loader(self):
    stream = torch.cuda.Stream()
    first = True

    for next_inputs, next_targets in self.dataloader:
      with torch.cuda.stream(stream):
        next_inputs = next_inputs.to(
            self.device, dtype=torch.float,
            non_blocking=True).sub(self.data_mean).div(self.data_std)
        next_targets = next_targets.to(self.device, non_blocking=True)

      if not first:
        yield inputs, targets
      else:
        first = False

      torch.cuda.current_stream().wait_stream(stream)
      inputs = next_inputs
      targets = next_targets

    yield inputs, targets


# Inspired by github.com/PetrochukM/PyTorch-NLP/blob/master/torchnlp/samplers/
# distributed_sampler.py
class TFDistributedSampler:

  def __init__(self, iterator, device='cuda:0', rank=None):
    self.iterator = iterator
    self.device = device
    self.rank = rank
    if rank is None:
      if not torch.distributed.is_initialized():
        raise RuntimeError('Requires `torch.distributed` to be initialized.')
      self.rank = torch.distributed.get_rank()

  def __iter__(self):
    return self

  def __next__(self):
    batch = next(self.iterator)
    batch = {
        # Assumes that len(value) > self.rank, i.e. there needs to be data for
        # each rank/GPU.
        key: torch.as_tensor(
            value[self.rank], device=self.device, dtype=torch.int64) for key,
        value in batch.items()
    }
    return batch
