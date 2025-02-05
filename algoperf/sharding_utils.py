"""Utilities for dealing with sharding in JAX."""

import jax
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec


def get_mesh() -> jax.sharding.Mesh:
  """Creates a mesh from all available GPUs. Here, we simply create a one-dimensional mesh."""
  return jax.sharding.Mesh(jax.devices(), ("batch",))


def get_replicated_sharding(mesh=None):
  """Returns a sharding spec that replicates data across all devices."""
  if mesh is None:
    mesh = get_mesh()
  return NamedSharding(mesh, PartitionSpec())


def get_naive_sharding_spec(mesh=None):
  """Returns a sharding spec that shards data along the first axis."""
  if mesh is None:
    mesh = get_mesh()
  return NamedSharding(mesh, PartitionSpec("batch"))


def get_naive_sharding(x, mesh=None):
  """Given a 1D mesh and a tensor, try to shard along the appropriate axis."""
  if mesh is None:
    mesh = get_mesh()
  grid_size = mesh.shape["batch"]
  if x.shape[0] % grid_size == 0:
    return NamedSharding(mesh, PartitionSpec("batch"))
  else:
    return NamedSharding(mesh, PartitionSpec())


def shard_params(params, mesh=None):
  """Shards a parameter tree across all devices with naive sharding (see get_naive_sharding)."""
  if mesh is None:
    mesh = get_mesh()
  return jax.tree_util.tree_map(
      lambda x: jax.device_put(x, get_naive_sharding(x)), params)


def get_sharding_tree(params, mesh=None):
  """Returns a sharding tree for a parameter tree."""
  return jax.tree_util.tree_map(lambda x: get_naive_sharding(x, mesh), params)


def get_empty_sharding(mesh=None):
  """Returns a sharding spec that replicates data across all devices."""
  if mesh is None:
    mesh = get_mesh()
  return NamedSharding(mesh, PartitionSpec())


def disp_shard_info(x: jax.Array):
  """Displays shard info of a jax array."""
  for shard in x.addressable_shards:
    print(f"shard.device: {shard.device}, index: {shard.index}, replica_id:"
          f" {shard.replica_id}.\n")
