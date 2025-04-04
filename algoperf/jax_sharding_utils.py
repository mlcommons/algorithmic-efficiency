"""Utilities for dealing with sharding in JAX."""

import jax
from jax.sharding import NamedSharding, PartitionSpec as P


def get_replicate_sharding():
  """Returns a sharding spec that replicates data across all devices."""
  mesh = jax.sharding.Mesh(jax.devices(), ('batch',))
  return NamedSharding(mesh, P())


def get_batch_dim_sharding():
  """Returns a sharding spec that shards data along the first axis."""
  mesh = jax.sharding.Mesh(jax.devices(), ('batch',))
  return NamedSharding(mesh, P('batch'))


def shard_along_batch_dim(x):
  """Shards a tensor across all devices."""
  mesh = jax.sharding.Mesh(jax.devices(), ('batch',))
  return jax.tree.map(
      lambda x: jax.device_put(x, NamedSharding(mesh, P('batch'))))


def replicate(x):
  """Replicates tensor across all devices."""
  mesh = jax.sharding.Mesh(jax.devices(), ('batch',))
  return jax.tree.map(
      lambda x: jax.device_put(x, NamedSharding(mesh, P())), x)


def disp_shard_info(x: jax.Array):
  """Displays shard info of a jax array."""
  for shard in x.addressable_shards:
    print(f"shard.device: {shard.device}, index: {shard.index}, replica_id:"
          f" {shard.replica_id}.\n")