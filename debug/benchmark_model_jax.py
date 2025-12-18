"""Benchmark script for JAX ImageNet ResNet50 model (forward + backward)."""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import pop

from algoperf import jax_sharding_utils
from algoperf.workloads.imagenet_resnet.imagenet_jax import models

# Training config
BATCH_SIZE = 1024
IMAGE_SIZE = 224
NUM_CLASSES = 1000
NUM_WARMUP = 10
NUM_BENCHMARK = 100


def main():
  print(f'=== JAX ResNet50 Model Benchmark ===')
  print(f'Batch size: {BATCH_SIZE}')
  print(f'Image size: {IMAGE_SIZE}')
  print(f'Num devices: {jax.local_device_count()}')
  print(f'Devices: {jax.devices()}')

  # Initialize model
  print('\nInitializing model...')
  rng = jax.random.PRNGKey(0)
  model = models.ResNet50(num_classes=NUM_CLASSES, act=nn.relu, dtype=jnp.float32)

  input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)
  variables = model.init({'params': rng}, jnp.ones(input_shape, jnp.float32))
  model_state, params = pop(variables, 'params')

  # Replicate params and model_state across devices (like the workload does)
  params = jax.tree.map(
    lambda x: jax.device_put(x, jax_sharding_utils.get_replicate_sharding()),
    params,
  )
  model_state = jax.tree.map(
    lambda x: jax.device_put(x, jax_sharding_utils.get_replicate_sharding()),
    model_state,
  )

  print(f'Model initialized. Param count: {sum(p.size for p in jax.tree.leaves(params)):,}')

  # Define forward pass (jit compiled with sharding)
  @functools.partial(
    jax.jit,
    in_shardings=(
      jax_sharding_utils.get_replicate_sharding(),  # params
      jax_sharding_utils.get_replicate_sharding(),  # model_state
      jax_sharding_utils.get_batch_dim_sharding(),  # inputs
    ),
    out_shardings=(
      jax_sharding_utils.get_batch_dim_sharding(),  # logits
      jax_sharding_utils.get_replicate_sharding(),  # new_model_state
    ),
  )
  def forward_fn(params, model_state, inputs):
    variables = {'params': params, **model_state}
    logits, new_model_state = model.apply(
      variables,
      inputs,
      update_batch_norm=True,
      mutable=['batch_stats'],
    )
    return logits, new_model_state

  # Define loss function (called inside train_step, not jitted separately)
  def loss_fn(params, model_state, inputs, targets):
    variables = {'params': params, **model_state}
    logits, new_model_state = model.apply(
      variables,
      inputs,
      update_batch_norm=True,
      mutable=['batch_stats'],
    )
    one_hot_targets = jax.nn.one_hot(targets, NUM_CLASSES)
    per_example_loss = -jnp.sum(one_hot_targets * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    loss = jnp.mean(per_example_loss)
    return loss, new_model_state

  # Define forward + backward pass (jit compiled with sharding)
  @functools.partial(
    jax.jit,
    in_shardings=(
      jax_sharding_utils.get_replicate_sharding(),  # params
      jax_sharding_utils.get_replicate_sharding(),  # model_state
      jax_sharding_utils.get_batch_dim_sharding(),  # inputs
      jax_sharding_utils.get_batch_dim_sharding(),  # targets
    ),
    out_shardings=(
      jax_sharding_utils.get_replicate_sharding(),  # loss
      jax_sharding_utils.get_replicate_sharding(),  # grads
      jax_sharding_utils.get_replicate_sharding(),  # new_model_state
    ),
  )
  def train_step(params, model_state, inputs, targets):
    (loss, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
      params, model_state, inputs, targets
    )
    return loss, grads, new_model_state

  # Generate random data and shard along batch dimension
  print('Generating random data...')
  data_rng = jax.random.PRNGKey(42)
  inputs = jax.random.normal(data_rng, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=jnp.float32)
  targets = jax.random.randint(jax.random.PRNGKey(43), (BATCH_SIZE,), 0, NUM_CLASSES)

  # Shard inputs along batch dimension
  inputs = jax.device_put(inputs, jax_sharding_utils.get_batch_dim_sharding())
  targets = jax.device_put(targets, jax_sharding_utils.get_batch_dim_sharding())

  print(f'Input sharding: {inputs.sharding}')

  # Warmup forward pass
  print(f'\nWarming up forward pass ({NUM_WARMUP} iterations)...')
  for i in range(NUM_WARMUP):
    start = time.perf_counter()
    logits, _ = forward_fn(params, model_state, inputs)
    logits.block_until_ready()
    end = time.perf_counter()
    print(f'  Warmup {i+1}/{NUM_WARMUP}: {(end - start)*1000:.2f}ms')

  # Benchmark forward pass
  print(f'\nBenchmarking forward pass ({NUM_BENCHMARK} iterations)...')
  forward_times = []
  for i in range(NUM_BENCHMARK):
    start = time.perf_counter()
    logits, _ = forward_fn(params, model_state, inputs)
    logits.block_until_ready()
    end = time.perf_counter()
    forward_times.append(end - start)
    if (i + 1) % 20 == 0:
      print(f'  Batch {i+1}/{NUM_BENCHMARK}: {forward_times[-1]*1000:.2f}ms')

  forward_times = np.array(forward_times)
  print(f'\n--- Forward Pass Results ---')
  print(f'Mean: {forward_times.mean()*1000:.2f}ms')
  print(f'Std: {forward_times.std()*1000:.2f}ms')
  print(f'Min: {forward_times.min()*1000:.2f}ms')
  print(f'Max: {forward_times.max()*1000:.2f}ms')
  print(f'Throughput: {BATCH_SIZE / forward_times.mean():.2f} images/sec')

  # Warmup forward + backward pass
  print(f'\nWarming up forward+backward pass ({NUM_WARMUP} iterations)...')
  for i in range(NUM_WARMUP):
    start = time.perf_counter()
    loss, grads, new_model_state = train_step(params, model_state, inputs, targets)
    loss.block_until_ready()
    end = time.perf_counter()
    print(f'  Warmup {i+1}/{NUM_WARMUP}: {(end - start)*1000:.2f}ms')

  # Benchmark forward + backward pass
  print(f'\nBenchmarking forward+backward pass ({NUM_BENCHMARK} iterations)...')
  train_times = []
  for i in range(NUM_BENCHMARK):
    start = time.perf_counter()
    loss, grads, new_model_state = train_step(params, model_state, inputs, targets)
    loss.block_until_ready()
    end = time.perf_counter()
    train_times.append(end - start)
    if (i + 1) % 20 == 0:
      print(f'  Batch {i+1}/{NUM_BENCHMARK}: {train_times[-1]*1000:.2f}ms')

  train_times = np.array(train_times)
  print(f'\n--- Forward+Backward Pass Results ---')
  print(f'Mean: {train_times.mean()*1000:.2f}ms')
  print(f'Std: {train_times.std()*1000:.2f}ms')
  print(f'Min: {train_times.min()*1000:.2f}ms')
  print(f'Max: {train_times.max()*1000:.2f}ms')
  print(f'Throughput: {BATCH_SIZE / train_times.mean():.2f} images/sec')

  # Print machine-readable results for the fish script
  print(f'\n=== RESULTS ===')
  print(f'FORWARD_MEAN_MS={forward_times.mean()*1000:.2f}')
  print(f'FORWARD_THROUGHPUT={BATCH_SIZE / forward_times.mean():.2f}')
  print(f'TRAIN_MEAN_MS={train_times.mean()*1000:.2f}')
  print(f'TRAIN_THROUGHPUT={BATCH_SIZE / train_times.mean():.2f}')


if __name__ == '__main__':
  main()
