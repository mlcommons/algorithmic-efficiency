"""Tests for LM HuggingFace input pipeline."""
import os

import jax
import jax.numpy as jnp
import torch
from transformers import GPT2Tokenizer

from algoperf.workloads.lm.input_pipeline import get_hf_dataloader


def main():
  # Setup test environment
  cache_dir = "/home/ak4605/data"
  if not os.path.exists(cache_dir):
    raise FileNotFoundError(f"Cache directory {cache_dir} not found")

  data_rng = jax.random.PRNGKey(42)
  tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
  vocab_size = tokenizer.vocab_size

  print("Running JAX output shapes and types test...")
  batch_size = 8
  seq_len = 32
  loader = get_hf_dataloader(
      cache_dir=cache_dir,
      batch_size=batch_size,
      seq_len=seq_len,
      framework="jax",
      split="train",
      data_rng=data_rng)
  inputs, targets = next(loader)
  assert inputs.shape == (batch_size, seq_len, vocab_size), \
      f"Expected inputs shape {(batch_size, seq_len, vocab_size)}, got {inputs.shape}"
  assert targets.shape == (batch_size, seq_len, vocab_size), \
      f"Expected targets shape {(batch_size, seq_len, vocab_size)}, got {targets.shape}"
  assert inputs.dtype == jnp.float32, \
      f"Expected inputs dtype float32, got {inputs.dtype}"
  assert targets.dtype == jnp.float32, \
      f"Expected targets dtype float32, got {targets.dtype}"
  assert jnp.all(jnp.sum(inputs, axis=-1) == 1), "Inputs should be one-hot encoded"
  assert jnp.all(jnp.sum(targets, axis=-1) == 1), "Targets should be one-hot encoded"
  print("✓ JAX test passed")

  print("\nRunning Torch output shapes and types test...")
  loader = get_hf_dataloader(
      cache_dir=cache_dir,
      batch_size=batch_size,
      seq_len=seq_len,
      framework="torch",
      split="train",
      data_rng=data_rng)
  inputs, targets = next(loader)
  assert inputs.shape == (batch_size, seq_len, vocab_size), \
      f"Expected inputs shape {(batch_size, seq_len, vocab_size)}, got {inputs.shape}"
  assert targets.shape == (batch_size, seq_len, vocab_size), \
      f"Expected targets shape {(batch_size, seq_len, vocab_size)}, got {targets.shape}"
  assert inputs.dtype == torch.float32, \
      f"Expected inputs dtype float32, got {inputs.dtype}"
  assert targets.dtype == torch.float32, \
      f"Expected targets dtype float32, got {targets.dtype}"
  assert torch.all(torch.sum(inputs, dim=-1) == 1), "Inputs should be one-hot encoded"
  assert torch.all(torch.sum(targets, dim=-1) == 1), "Targets should be one-hot encoded"
  print("✓ Torch test passed")

  print("\nTesting consistent batching with same seed...")
  loader1 = get_hf_dataloader(
      cache_dir=cache_dir,
      batch_size=batch_size,
      seq_len=seq_len,
      framework="jax",
      split="train",
      data_rng=jax.random.PRNGKey(42))
  batch1 = next(loader1)

  loader2 = get_hf_dataloader(
      cache_dir=cache_dir,
      batch_size=batch_size,
      seq_len=seq_len,
      framework="jax",
      split="train",
      data_rng=jax.random.PRNGKey(42))
  batch2 = next(loader2)

  assert jnp.array_equal(batch1[0], batch2[0]), "Input batches should be identical with same seed"
  assert jnp.array_equal(batch1[1], batch2[1]), "Target batches should be identical with same seed"
  print("✓ Consistent batching test passed")

  print("\nTesting eval split doesn't shuffle...")
  loader1 = get_hf_dataloader(
      cache_dir=cache_dir,
      batch_size=batch_size,
      seq_len=seq_len,
      framework="jax",
      split="eval",
      data_rng=jax.random.PRNGKey(42))
  batch1 = next(loader1)

  loader2 = get_hf_dataloader(
      cache_dir=cache_dir,
      batch_size=batch_size,
      seq_len=seq_len,
      framework="jax",
      split="eval",
      data_rng=jax.random.PRNGKey(999))
  batch2 = next(loader2)

  assert jnp.array_equal(batch1[0], batch2[0]), "Eval inputs should be identical regardless of seed"
  assert jnp.array_equal(batch1[1], batch2[1]), "Eval targets should be identical regardless of seed"
  print("✓ Eval no shuffling test passed")

  print("\nAll tests passed successfully!")


if __name__ == "__main__":
  main()
