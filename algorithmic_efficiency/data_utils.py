import jax


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
      yield {key: value for key, value in zip(keys, batch)}
    except StopIteration:
      if custom_sampler:
        epoch += 1
        iterable.sampler.set_epoch(epoch)
      iterator = iter(iterable)
