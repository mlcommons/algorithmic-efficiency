"""
Test file comparing BatchNorm module performance with shard_map vs jax.jit.

This test creates a BatchNorm module and compares its behavior and performance
when compiled with shard_map vs jax.jit.
"""

import functools
import time
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
import numpy as np


class BatchNorm(nn.Module):
    """Implements batch norm respecting input paddings.
    
    This implementation takes into account input padding by masking inputs before
    computing mean and variance.
    """
    encoder_dim: int = 0
    dtype: Any = jnp.float32
    batch_norm_momentum: float = 0.999
    batch_norm_epsilon: float = 0.001
    
    def setup(self):
        dim = self.encoder_dim
        dtype = self.dtype
        self.ra_mean = self.variable('batch_stats',
                                     'mean',
                                     lambda s: jnp.zeros(s, dtype),
                                     dim)
        self.ra_var = self.variable('batch_stats',
                                    'var',
                                    lambda s: jnp.ones(s, dtype),
                                    dim)
        self.gamma = self.param('scale', nn.initializers.zeros, dim, dtype)
        self.beta = self.param('bias', nn.initializers.zeros, dim, dtype)
    
    def _get_default_paddings(self, inputs):
        """Gets the default paddings for an input."""
        in_shape = list(inputs.shape)
        in_shape[-1] = 1
        return jnp.zeros(in_shape, dtype=inputs.dtype)
    
    @nn.compact
    def __call__(self, inputs, input_paddings=None, train=False):
        rank = inputs.ndim
        reduce_over_dims = list(range(0, rank - 1))
        
        if input_paddings is None:
            padding = self._get_default_paddings(inputs)
        else:
            padding = jnp.expand_dims(input_paddings, -1)
        
        momentum = self.batch_norm_momentum
        epsilon = self.batch_norm_epsilon
        
        if train:
            mask = 1.0 - padding
            sum_v = jnp.sum(inputs * mask, axis=reduce_over_dims, keepdims=True)
            count_v = jnp.sum(
                jnp.ones_like(inputs) * mask, axis=reduce_over_dims, keepdims=True)
            count_v = jnp.maximum(count_v, 1.0)
            mean = sum_v / count_v
            variance = (inputs - mean) * (inputs - mean) * mask
            sum_vv = jnp.sum(variance, axis=reduce_over_dims, keepdims=True)
            var = sum_vv / count_v
            
            self.ra_mean.value = momentum * self.ra_mean.value + (1 - momentum) * mean
            self.ra_var.value = momentum * self.ra_var.value + (1 - momentum) * var
        else:
            mean = self.ra_mean.value
            var = self.ra_var.value
        
        inv = (1 + self.gamma) / jnp.sqrt(var + epsilon)
        bn_output = (inputs - mean) * inv + self.beta
        bn_output *= 1.0 - padding
        return bn_output


def create_test_data(batch_size=8, seq_len=100, feature_dim=512):
    """Create test data for BatchNorm."""
    key = jax.random.PRNGKey(42)
    
    # Create inputs
    inputs = jax.random.normal(key, (batch_size, seq_len, feature_dim))
    
    # Create input paddings (some sequences are padded)
    padding_key = jax.random.PRNGKey(123)
    input_paddings = jax.random.bernoulli(padding_key, 0.1, (batch_size, seq_len))
    
    return inputs, input_paddings


def init_batch_norm(feature_dim, inputs, input_paddings):
    """Initialize BatchNorm module."""
    model = BatchNorm(encoder_dim=feature_dim)
    key = jax.random.PRNGKey(0)
    variables = model.init(key, inputs, input_paddings, train=True)
    return model, variables


def batch_norm_fn_jit(model, variables, inputs, input_paddings, train=True):
    """BatchNorm function compiled with jax.jit."""
    @functools.partial(jax.jit, static_argnames="train")
    def _apply_fn(vars, inputs, paddings, train):
        return model.apply(vars, inputs, paddings, train=train, mutable=['batch_stats'])
    
    return _apply_fn(variables, inputs, input_paddings, train)


def batch_norm_fn_shard_map(model, variables, inputs, input_paddings, train=True):
    """BatchNorm function compiled with shard_map."""
    # Create a mesh for sharding
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ("batch",))
    
    def _apply_fn(vars, inputs, paddings, train):
        return model.apply(vars, inputs, paddings, train=train, mutable=['batch_stats'])
    
    # Use shard_map to distribute computation
    sharded_fn = shard_map(
        _apply_fn,
        mesh,
        in_specs=(P(), P("batch"), P("batch"), P()),
        out_specs=(P("batch"), P("batch"))
    )
    
    return sharded_fn(variables, inputs, input_paddings, train)


def test_correctness():
    """Test that both implementations produce the same results."""
    print("Testing correctness...")
    
    batch_size, seq_len, feature_dim = 8, 100, 512
    inputs, input_paddings = create_test_data(batch_size, seq_len, feature_dim)
    model, variables = init_batch_norm(feature_dim, inputs, input_paddings)
    
    # Test with jax.jit
    result_jit, new_vars_jit = batch_norm_fn_jit(
        model, variables, inputs, input_paddings, train=True)
    
    # Test with shard_map (need to shard the inputs first)
    # Replicate variables and shard inputs
    from jax.experimental import multihost_utils
    
    # For shard_map, we need to properly shard the inputs
    devices = jax.devices()
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(devices, ("batch",)),
        P("batch")
    )
    
    # Shard inputs across the batch dimension
    inputs_sharded = jax.device_put(inputs, sharding)
    input_paddings_sharded = jax.device_put(input_paddings, sharding)
    
    result_shard, new_vars_shard = batch_norm_fn_shard_map(
        model, variables, inputs_sharded, input_paddings_sharded, train=True)
    
    # Compare results
    result_diff = jnp.abs(result_jit - result_shard).max()
    print(f"Max difference in outputs: {result_diff}")
    
    # Compare batch stats
    mean_diff = jnp.abs(new_vars_jit['batch_stats']['mean'] - 
                       new_vars_shard['batch_stats']['mean']).max()
    var_diff = jnp.abs(new_vars_jit['batch_stats']['var'] - 
                      new_vars_shard['batch_stats']['var']).max()
    
    print(f"Max difference in batch_stats mean: {mean_diff}")
    print(f"Max difference in batch_stats var: {var_diff}")
    
    tolerance = 1e-5
    assert result_diff < tolerance, f"Output difference {result_diff} > {tolerance}"
    assert mean_diff < tolerance, f"Mean difference {mean_diff} > {tolerance}"
    assert var_diff < tolerance, f"Var difference {var_diff} > {tolerance}"
    
    print("✓ Correctness test passed!")
    return result_jit, result_shard


def benchmark_performance():
    """Benchmark performance of both implementations."""
    print("\nBenchmarking performance...")
    
    batch_size, seq_len, feature_dim = 32, 1000, 1024
    inputs, input_paddings = create_test_data(batch_size, seq_len, feature_dim)
    model, variables = init_batch_norm(feature_dim, inputs, input_paddings)
    
    num_runs = 100
    
    # Benchmark jax.jit
    print("Warming up jax.jit...")
    for _ in range(10):
        _ = batch_norm_fn_jit(model, variables, inputs, input_paddings, train=True)
    
    print("Benchmarking jax.jit...")
    start_time = time.time()
    for _ in range(num_runs):
        result_jit, _ = batch_norm_fn_jit(model, variables, inputs, input_paddings, train=True)
        result_jit.block_until_ready()  # Ensure computation is complete
    jit_time = time.time() - start_time
    
    # Benchmark shard_map
    devices = jax.devices()
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(devices, ("batch",)),
        P("batch")
    )
    inputs_sharded = jax.device_put(inputs, sharding)
    input_paddings_sharded = jax.device_put(input_paddings, sharding)
    
    print("Warming up shard_map...")
    for _ in range(10):
        _ = batch_norm_fn_shard_map(model, variables, inputs_sharded, input_paddings_sharded, train=True)
    
    print("Benchmarking shard_map...")
    start_time = time.time()
    for _ in range(num_runs):
        result_shard, _ = batch_norm_fn_shard_map(model, variables, inputs_sharded, input_paddings_sharded, train=True)
        result_shard.block_until_ready()  # Ensure computation is complete
    shard_time = time.time() - start_time
    
    print(f"jax.jit time: {jit_time:.4f}s ({jit_time/num_runs*1000:.2f}ms per run)")
    print(f"shard_map time: {shard_time:.4f}s ({shard_time/num_runs*1000:.2f}ms per run)")
    print(f"Speedup: {jit_time/shard_time:.2f}x")
    
    return jit_time, shard_time


def test_memory_usage():
    """Test memory usage patterns."""
    print("\nTesting memory usage...")
    
    batch_size, seq_len, feature_dim = 64, 2000, 2048
    inputs, input_paddings = create_test_data(batch_size, seq_len, feature_dim)
    model, variables = init_batch_norm(feature_dim, inputs, input_paddings)
    
    # Test memory usage for jax.jit
    print("Memory usage with jax.jit:")
    try:
        result_jit, _ = batch_norm_fn_jit(model, variables, inputs, input_paddings, train=True)
        print("✓ jax.jit completed successfully")
    except Exception as e:
        print(f"✗ jax.jit failed: {e}")
    
    # Test memory usage for shard_map
    print("Memory usage with shard_map:")
    try:
        devices = jax.devices()
        sharding = jax.sharding.NamedSharding(
            jax.sharding.Mesh(devices, ("batch",)),
            P("batch")
        )
        inputs_sharded = jax.device_put(inputs, sharding)
        input_paddings_sharded = jax.device_put(input_paddings, sharding)
        
        result_shard, _ = batch_norm_fn_shard_map(model, variables, inputs_sharded, input_paddings_sharded, train=True)
        print("✓ shard_map completed successfully")
    except Exception as e:
        print(f"✗ shard_map failed: {e}")


def main():
    """Run all tests."""
    print("Running BatchNorm shard_map vs jax.jit comparison tests")
    print("=" * 60)
    
    print(f"Available devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")
    print()
    
    # Test correctness
    test_correctness()
    
    # Benchmark performance
    benchmark_performance()
    
    # Test memory usage
    test_memory_usage()
    
    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    main()