"""
Originally based on code from the NanoDO repository under the Apache 2.0 license:
https://github.com/google-deepmind/nanodo
"""

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import jmp
from flax import linen as nn


@dataclasses.dataclass
class ModelConfig:
  """Hyper-parameters for Transformer decoder-only."""

  model_dim: int  # model/embed dim = qkv dim
  num_heads: int  # num attention heads
  seq_len: int  # max context/sequence length
  num_layers: int  # number of transformer block layers
  vocab_size: int  # vocab size
  expanded_model_dim: int  # FF inner dimension
  multiple_of: int = 256
  rmsnorm_epsilon: float = 1e-6
  use_residual_scaling: bool = True
  tie_embeddings: bool = True  # Whether to tie input and output embed
  qknorm_epsilon: float = 1e-6
  attention_init: nn.initializers.Initializer = nn.initializers.normal(
    stddev=0.02
  )
  linear_init: nn.initializers.Initializer = nn.initializers.normal(stddev=0.02)
  embed_init: nn.initializers.Initializer = nn.initializers.normal(stddev=0.02)
  param_dtype: jnp.dtype = jnp.float32
  compute_dtype: jnp.dtype = jnp.bfloat16
  output_dtype: jnp.dtype = jnp.bfloat16

  def __post_init__(self):
    self.residual_init = nn.initializers.normal(
      stddev=0.02 / jnp.sqrt(2 * self.num_layers)
    )
    self.mp_policy = jmp.Policy(
      compute_dtype=self.compute_dtype,
      param_dtype=self.param_dtype,
      output_dtype=self.output_dtype,
    )


class Mlp(nn.Module):
  """Multilayer perceptron with GLU activation."""

  cfg: ModelConfig

  @nn.compact
  def __call__(self, x_BxLxD: jax.Array):
    cfg = self.cfg
    linear = partial(
      nn.Dense,
      kernel_init=cfg.linear_init,
      use_bias=False,
      dtype=cfg.compute_dtype,
      param_dtype=cfg.param_dtype,
    )
    #  Adjust hidden dimension to keep the number of parameters invariant to
    # the activation function used since the GLU MLP has 3 * hidden_dim * D
    # parameters instead of 2 * hidden_dim * D parameters
    hidden_dim = cfg.expanded_model_dim * 2 / 3
    hidden_dim = cfg.multiple_of * (
      (cfg.expanded_model_dim + cfg.multiple_of - 1) // cfg.multiple_of
    )
    # Double the hidden dimension for GLU
    x_BxLx2F = linear(2 * hidden_dim)(x_BxLxD)
    # Apply GLU activation
    x_BxLxF = nn.glu(x_BxLx2F, axis=-1)
    x_BxLxD = nn.Dense(
      cfg.model_dim,
      use_bias=False,
      dtype=cfg.compute_dtype,
      param_dtype=cfg.param_dtype,
      kernel_init=cfg.residual_init
      if cfg.use_residual_scaling
      else cfg.linear_init,
    )(x_BxLxF)
    return x_BxLxD


@partial(jax.jit, static_argnums=(0, 1, 2))
def init_rope(dim=256, seq_len=128, n_heads=4):
  """Initialize rotary embeddings."""

  def precompute_freqs_cis_jax(dim, end, theta=10000.0):
    inv_freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
    t = jnp.arange(end) / 1.0
    freqs = jnp.outer(t, inv_freqs).astype(jnp.float32)
    return jnp.stack(
      [jnp.cos(freqs)[None, :, None, :], jnp.sin(freqs)[None, :, None, :]],
      axis=3,
    )

  freqs_cis = precompute_freqs_cis_jax(dim // n_heads, seq_len, theta=500000)
  return freqs_cis.transpose(0, 1, 2, 4, 3)


@jax.jit
def apply_rope(q, k, freqs_cis):
  """Apply rotary embeddings to Q and K."""

  def rotate_tensor(x):
    # Split into real and imaginary parts
    x_r2 = x.reshape(*x.shape[:-1], -1, 2).astype(jnp.float32)
    L = x.shape[1]
    freqs = freqs_cis[:, :L, :, :, :]

    # Apply rotation
    rotated_x_r2 = jnp.stack(
      [
        x_r2[..., 0] * freqs[..., 0] - x_r2[..., 1] * freqs[..., 1],
        x_r2[..., 1] * freqs[..., 0] + x_r2[..., 0] * freqs[..., 1],
      ],
      axis=-1,
    )

    return rotated_x_r2.reshape(*x.shape).astype(x.dtype)

  # Apply rotation to Q and K separately
  rotated_q = rotate_tensor(q)
  rotated_k = rotate_tensor(k)

  return rotated_q, rotated_k


class CausalAttn(nn.Module):
  """Causal attention layer with rotary embeddings."""

  cfg: ModelConfig

  def setup(self):
    cfg = self.cfg
    assert cfg.model_dim % cfg.num_heads == 0, (
      f'D {cfg.model_dim} not divisible by H {cfg.num_heads}'
    )
    self.Dh = cfg.model_dim // cfg.num_heads
    self.eps = cfg.qknorm_epsilon

    # Initialize rotary embeddings
    self.freqs_cis = init_rope(cfg.model_dim, cfg.seq_len, cfg.num_heads)

    # Maps D -> (H, Dh)
    self.multilinear = partial(
      nn.DenseGeneral,
      axis=-1,
      features=(cfg.num_heads, self.Dh),
      kernel_init=cfg.attention_init,
      use_bias=False,
      dtype=cfg.compute_dtype,
      param_dtype=cfg.param_dtype,
    )
    self.multilinear_query = self.multilinear(name='query')
    self.multilinear_key = self.multilinear(name='key')
    self.multilinear_value = self.multilinear(name='value')
    # See Henry et al. (2020) "Query Key Normalization for Transformers"
    seq_len = cfg.seq_len
    attn_scale0 = jnp.log2(seq_len**2 - seq_len)
    self.attn_scale = self.param(
      'attn_scale',
      nn.initializers.constant(attn_scale0, dtype=cfg.compute_dtype),
      (),
    )
    self.output_projection = nn.DenseGeneral(
      features=cfg.model_dim,
      name='attn_out_proj',
      # axis=(-2, -1),      #
      kernel_init=cfg.residual_init
      if cfg.use_residual_scaling
      else cfg.linear_init,
      use_bias=False,
      dtype=cfg.compute_dtype,
      param_dtype=cfg.param_dtype,
    )

  def __call__(self, x_BxLxD: jax.Array):
    cfg = self.cfg

    # Project inputs to Q, K, V
    q_BxLxHxDh = self.multilinear_query(x_BxLxD)
    k_BxLxHxDh = self.multilinear_key(x_BxLxD)
    v_BxLxHxDh = self.multilinear_value(x_BxLxD)

    # Apply rotary embeddings to Q and K
    q_BxLxHxDh, k_BxLxHxDh = apply_rope(q_BxLxHxDh, k_BxLxHxDh, self.freqs_cis)

    # Apply QK normalization
    q_BxLxHxDh /= jnp.linalg.norm(q_BxLxHxDh, axis=-1, keepdims=True) + self.eps
    k_BxLxHxDh /= jnp.linalg.norm(k_BxLxHxDh, axis=-1, keepdims=True) + self.eps
    q_BxLxHxDh *= self.attn_scale
    out_BxLxHxDh = jax.nn.dot_product_attention(
      query=q_BxLxHxDh,
      key=k_BxLxHxDh,
      value=v_BxLxHxDh,
      is_causal=True,
      scale=1.0,
      implementation='cudnn' if cfg.compute_dtype is not jnp.float32 else None,
    )
    out_BxLxD = out_BxLxHxDh.reshape(*x_BxLxD.shape)
    out_BxLxD = self.output_projection(out_BxLxD)
    return out_BxLxD


class TBlock(nn.Module):
  """Transformer Block."""

  docfg: ModelConfig

  @nn.compact
  def __call__(self, in_BxLxD: jax.Array):
    cfg = self.docfg

    # x = x + attn( attn_norm(x) )
    x_BxLxD = nn.RMSNorm(
      param_dtype=cfg.param_dtype, epsilon=cfg.rmsnorm_epsilon
    )(in_BxLxD)
    x_BxLxD = CausalAttn(cfg)(x_BxLxD)
    x_BxLxD += in_BxLxD

    # x = x + mlp( mlp_norm(x) )
    z_BxLxD = nn.RMSNorm(
      param_dtype=cfg.param_dtype, epsilon=cfg.rmsnorm_epsilon
    )(x_BxLxD)
    z_BxLxD = Mlp(cfg)(z_BxLxD)

    return x_BxLxD + z_BxLxD


class TransformerDo(nn.Module):
  """Transformer decoder-only."""

  docfg: ModelConfig

  def setup(self):
    cfg = self.docfg
    self.embed = nn.Embed(
      num_embeddings=cfg.vocab_size,
      features=cfg.model_dim,
      embedding_init=cfg.embed_init,
      dtype=cfg.compute_dtype,
      param_dtype=cfg.param_dtype,
    )

    self.blocks = [TBlock(cfg) for _ in range(cfg.num_layers)]
    self.out_ln = nn.RMSNorm(
      param_dtype=cfg.param_dtype, epsilon=cfg.rmsnorm_epsilon
    )

    # Output projection - tied to input embeddings if configured
    if cfg.tie_embeddings:
      self.output_proj = lambda x: self.embed.attend(x)
    else:
      self.output_proj = nn.Dense(
        cfg.vocab_size,
        kernel_init=cfg.embed_init,
        dtype=cfg.compute_dtype,
        param_dtype=cfg.param_dtype,
        name='output_proj',
      )

  def __call__(self, y_BxL: jax.Array):
    # For training on concatenated examples.
    y_BxLxD = self.embed(y_BxL)
    for block in self.blocks:
      y_BxLxD = block(y_BxLxD)
    y_BxLxD = self.out_ln(y_BxLxD)
    logits_BxLxV = self.output_proj(y_BxLxD)
    return logits_BxLxV

  def predict(self, y_BxL: jax.Array, k: int = 1):
    """Generate k tokens autoregressively.

    Args:
        y_BxL: Input token sequence of shape (batch_size, seq_len)
        k: Number of tokens to predict

    Returns:
        Tuple of (input_ids, predicted_ids)
    """
    cfg = self.docfg
    seq_len = y_BxL.shape[1]

    # Store original input
    original_input = y_BxL

    # Make sure we don't exceed the model's context length
    if seq_len + k > cfg.seq_len:
      raise ValueError(
        f"Total sequence length ({seq_len + k}) exceeds model's context length ({cfg.seq_len})"
      )

    # Generate k tokens autoregressively
    for _ in range(k):
      # Get logits for the entire sequence
      logits = self(y_BxL)

      # Get the logits for the last token in each sequence
      next_token_logits = logits[:, -1, :]
      last_token_id = y_BxL[:, -1]
      # Prevent predicting the same token consecutively
      next_token_logits = next_token_logits.at[
        jnp.arange(len(last_token_id)), last_token_id
      ].set(float('-inf'))

      # Get the most likely token
      next_token = jnp.argmax(next_token_logits, axis=-1)

      # Append the predicted token to the sequence
      y_BxL = jnp.concatenate([y_BxL, next_token[:, None]], axis=1)

    # Return original input and the k predicted tokens
    return original_input, y_BxL[:, -k:]


# =========== Demo Code ==========


def main():
  """Create and run the DecoderOnly Transformer model."""
  # Initialize model configuration with smaller parameters for demo
  B, L = (2, 128)  # Batch size, sequence length
  cfg = ModelConfig(
    model_dim=128,
    num_heads=4,
    seq_len=L,
    num_layers=2,
    vocab_size=256,
    expanded_model_dim=4 * 128,
  )
  model = TransformerDo(cfg)

  # Print model info
  print('\nModel Configuration:')
  print(f'  - Model dimension (D): {cfg.model_dim}')
  print(f'  - Number of heads (H): {cfg.num_heads}')
  print(f'  - Max sequence length (L): {cfg.seq_len}')
  print(f'  - Number of layers (N): {cfg.num_layers}')
  print(f'  - Vocabulary size (V): {cfg.vocab_size}')
  print(f'  - Feed forward dimension (F): {cfg.expanded_model_dim}')

  # Create random input tokens (simulated token IDs)
  rng_key = jax.random.PRNGKey(42)
  input_rng, init_rng = jax.random.split(rng_key)

  # Generate random token IDs (integers between 0 and vocab_size-1)
  x_BxL = jax.random.randint(
    input_rng, shape=(B, L), minval=0, maxval=cfg.vocab_size, dtype=jnp.int32
  )

  # Initialize model parameters
  print('\nInitializing model parameters...')
  params = model.init(init_rng, x_BxL)

  # Print parameter count
  param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
  print(f'Total parameters: {param_count:,}')

  # Make a prediction (forward pass)
  print('\nRunning forward pass...')
  params, x_BxL = cfg.mp_policy.cast_to_compute((params, x_BxL))
  logits = model.apply(params, x_BxL)

  # Print output shape and sample values
  print(
    f'\nOutput shape: {logits.shape} (batch_size, sequence_length, vocab_size)'
  )
  print(f'Output data type: {logits.dtype}')

  # Print sample logits (first 5 positions of the first sequence)
  print('\nSample logits (first sequence, first 5 positions, first 5 values):')
  for position in range(min(5, L)):
    print(f'  Position {position}: {logits[0, position, :5]}')

  # Get predictions (token with highest logit at each position)
  predictions = jnp.argmax(logits, axis=-1)
  print('\nPredicted token IDs (first sequence, first 10 positions):')
  print(predictions[0, :10])

  # Test the predict function
  print('\nTesting predict function...')
  # Use a shorter
  short_seq = x_BxL[:, :10]
  print(f'Input sequence shape: {short_seq.shape}')

  # Predict 5 tokens
  k = 5
  original, predicted = model.apply(params, short_seq, k, method=model.predict)

  # Get predictions (token with highest logit at each position)
  predictions = jnp.argmax(logits, axis=-1)
  print('\nPredicted token IDs (first sequence, first 10 positions):')
  print(predictions[0, :10])

  print('\nDone!')


if __name__ == '__main__':
  main()
