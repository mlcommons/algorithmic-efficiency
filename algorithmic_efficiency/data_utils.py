import jax
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def shard_numpy_ds(xs):
  """Prepare tf data for JAX

  Convert an input batch from tf Tensors to numpy arrays and reshape it to be
  sharded across devices.
  """
  local_device_count = jax.local_device_count()

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
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
      if custom_sampler:
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
  process can pass a :class`~torch.utils.data.DistributedSampler` instance as
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
