# Self-contained version of the DecoderOnly Transformer from NanoDO

import dataclasses
from functools import partial

from flax import linen as nn
import jax
import jax.numpy as jnp

# =========== Transformer Decoder-only Model ==========



@dataclasses.dataclass
class DoConfig:
    """Hyper-parameters for Transformer decoder-only."""

    D: int  # model/embed dim = qkv dim
    H: int  # num attention heads
    L: int  # max context/sequence length
    N: int  # number of transformer block layers
    V: int  # vocab size
    F: int  # FF inner dimension
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    embed_init: nn.initializers.Initializer = nn.initializers.variance_scaling(
        1.0, "fan_in", "normal", out_axis=0
    )
    dtype: jnp.dtype = jnp.float32
    rmsnorm_epsilon: float = 1e-6
    multiple_of: int = 256
    tie_embeddings: bool = True  # Whether to tie input and output embeddings


class Mlp(nn.Module):
    """Multilayer perceptron with GLU activation."""

    cfg: DoConfig

    @nn.compact
    def __call__(self, x_BxLxD: jax.Array):
        cfg = self.cfg
        # Use Xavier uniform initialization explicitly
        xavier_init = nn.initializers.xavier_uniform()
        linear = partial(
            nn.Dense, kernel_init=xavier_init, use_bias=False, dtype=cfg.dtype
        )
        hidden_dim = cfg.multiple_of * (
            (cfg.F + cfg.multiple_of - 1) // cfg.multiple_of
        )
        # Double the hidden dimension for GLU
        x_BxLx2F = linear(2 * hidden_dim)(x_BxLxD)
        # Apply GLU activation
        x_BxLxF = nn.glu(x_BxLx2F, axis=-1)
        x_BxLxD = linear(cfg.D)(x_BxLxF)
        return x_BxLxD

@partial(jax.jit, static_argnums=(0,1,2))
def init_rope(dim=256, seq_len=128, n_heads=4):
    """Initialize rotary embeddings."""
    def precompute_freqs_cis_jax(dim, end, theta=10000.0):
        inv_freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))
        t = jnp.arange(end) / 1.0
        freqs = jnp.outer(t, inv_freqs).astype(jnp.float32)
        return jnp.stack([
            jnp.cos(freqs)[None, :, None, :],
            jnp.sin(freqs)[None, :, None, :]
        ], axis=3)

    freqs_cis = precompute_freqs_cis_jax(dim // n_heads, seq_len, theta=500000)
    return freqs_cis.transpose(0, 1, 2, 4, 3)

@jax.jit
def apply_rope(q, k, freqs_cis):
    """Apply rotary embeddings to Q and K."""
    def rotate_tensor(x):
        # Split into real and imaginary parts
        x_r2 = x.reshape(*x.shape[:-1], -1, 2)
        L = x.shape[1]
        freqs = freqs_cis[:, :L, :, :, :]

        # Apply rotation
        rotated_x_r2 = jnp.stack([
            x_r2[..., 0] * freqs[..., 0] - x_r2[..., 1] * freqs[..., 1],
            x_r2[..., 1] * freqs[..., 0] + x_r2[..., 0] * freqs[..., 1]
        ], axis=-1)

        return rotated_x_r2.reshape(*x.shape)

    # Apply rotation to Q and K separately
    rotated_q = rotate_tensor(q)
    rotated_k = rotate_tensor(k)

    return rotated_q, rotated_k


class CausalAttn(nn.Module):
    """Causal attention layer with rotary embeddings."""

    cfg: DoConfig

    def setup(self):
        cfg = self.cfg
        assert cfg.D % cfg.H == 0, f"D {cfg.D} not divisible by H {cfg.H}"
        self.Dh = cfg.D // cfg.H

        # Initialize rotary embeddings
        self.freqs_cis = init_rope(cfg.D, cfg.L, cfg.H)

        # Maps D -> (H, Dh)
        self.multilinear = partial(
            nn.DenseGeneral,
            axis=-1,
            features=(cfg.H, self.Dh),
            kernel_init=cfg.kernel_init,
            use_bias=False,
            dtype=cfg.dtype,
        )

        self.multilinear_query = self.multilinear(name="query")
        self.multilinear_key = self.multilinear(name="key")
        self.multilinear_value = self.multilinear(name="value")
        self.output_projection = nn.DenseGeneral(
            features=cfg.D,
            name="attn_out_proj",
            # axis=(-2, -1),      #
            kernel_init=cfg.kernel_init,
            use_bias=False,
            dtype=cfg.dtype,
        )

    def __call__(self, x_BxLxD: jax.Array):
        cfg = self.cfg

        # Project inputs to Q, K, V
        q_BxLxHxDh = self.multilinear_query(x_BxLxD)
        k_BxLxHxDh = self.multilinear_key(x_BxLxD)
        v_BxLxHxDh = self.multilinear_value(x_BxLxD)

        # Apply rotary embeddings to Q and K
        q_BxLxHxDh, k_BxLxHxDh = apply_rope(q_BxLxHxDh, k_BxLxHxDh, self.freqs_cis)

        # Scale queries
        q_BxLxHxDh /= self.Dh**0.5

        # Compute attention scores
        att_BxHxLxL = jnp.einsum("...qhd,...khd->...hqk", q_BxLxHxDh, k_BxLxHxDh)

        # Causal attention mask
        L = x_BxLxD.shape[1]
        mask_1x1xLxL = jnp.tril(jnp.ones((1, 1, L, L), dtype=jnp.bool_))

        # Apply mask and softmax
        _NEG_INF = jnp.finfo(cfg.dtype).min
        att_BxHxLxL = jnp.where(mask_1x1xLxL, att_BxHxLxL, _NEG_INF)
        att_BxHxLxL = jax.nn.softmax(att_BxHxLxL, axis=-1)
        att_BxHxLxL = att_BxHxLxL.astype(cfg.dtype)

        # Compute attention output
        out_BxLxHxDh = jnp.einsum("...hqk,...khd->...qhd", att_BxHxLxL, v_BxLxHxDh)

        # Reshape and project output
        out_BxLxD = out_BxLxHxDh.reshape(*x_BxLxD.shape)

        # Output projection
        out_BxLxD = self.output_projection(out_BxLxD)

        return out_BxLxD


class TBlock(nn.Module):
    """Transformer Block."""

    docfg: DoConfig

    @nn.compact
    def __call__(self, in_BxLxD: jax.Array):
        cfg = self.docfg

        # x = x + attn( attn_norm(x) )
        x_BxLxD = nn.RMSNorm(param_dtype=cfg.dtype, epsilon=cfg.rmsnorm_epsilon)(
            in_BxLxD
        )
        x_BxLxD = CausalAttn(cfg)(x_BxLxD)
        x_BxLxD += in_BxLxD

        # x = x + mlp( mlp_norm(x) )
        z_BxLxD = nn.RMSNorm(param_dtype=cfg.dtype, epsilon=cfg.rmsnorm_epsilon)(
            x_BxLxD
        )
        z_BxLxD = Mlp(cfg)(z_BxLxD)

        return x_BxLxD + z_BxLxD


class TransformerDo(nn.Module):
    """Transformer decoder-only."""

    docfg: DoConfig

    def setup(self):
        cfg = self.docfg
        self.embed = nn.Embed(
            num_embeddings=cfg.V,
            features=cfg.D,
            embedding_init=cfg.embed_init,
        )

        self.blocks = [TBlock(cfg) for _ in range(cfg.N)]
        self.out_ln = nn.RMSNorm(param_dtype=cfg.dtype, epsilon=cfg.rmsnorm_epsilon)

        # Output projection - tied to input embeddings if configured
        if cfg.tie_embeddings:
            self.output_proj = lambda x: self.embed.attend(x.astype(jnp.float32))
        else:
            self.output_proj = nn.Dense(
                cfg.V,
                kernel_init=cfg.embed_init,
                dtype=cfg.dtype,
                name="output_proj"
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
        batch_size = y_BxL.shape[0]
        seq_len = y_BxL.shape[1]

        # Store original input
        original_input = y_BxL

        # Make sure we don't exceed the model's context length
        if seq_len + k > cfg.L:
            raise ValueError(
                f"Total sequence length ({seq_len + k}) exceeds model's context length ({cfg.L})"
            )

        # Generate k tokens autoregressively
        for _ in range(k):
            # Get logits for the entire sequence
            logits = self(y_BxL)

            # Get the logits for the last token in each sequence
            next_token_logits = logits[:, -1, :]

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
    cfg = DoConfig(D=128, H=4, L=L, N=2, V=256, F=4 * 128)
    model = TransformerDo(cfg)

    # Print model info
    print(f"\nModel Configuration:")
    print(f"  - Model dimension (D): {cfg.D}")
    print(f"  - Number of heads (H): {cfg.H}")
    print(f"  - Max sequence length (L): {cfg.L}")
    print(f"  - Number of layers (N): {cfg.N}")
    print(f"  - Vocabulary size (V): {cfg.V}")
    print(f"  - Feed forward dimension (F): {cfg.F}")

    # Create random input tokens (simulated token IDs)
    rng_key = jax.random.PRNGKey(42)
    input_rng, init_rng = jax.random.split(rng_key)

    # Generate random token IDs (integers between 0 and vocab_size-1)
    x_BxL = jax.random.randint(
        input_rng, shape=(B, L), minval=0, maxval=cfg.V, dtype=jnp.int32
    )

    # Initialize model parameters
    print("\nInitializing model parameters...")
    params = model.init(init_rng, x_BxL)

    # Print parameter count
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {param_count:,}")

    # Make a prediction (forward pass)
    print("\nRunning forward pass...")
    logits = model.apply(params, x_BxL)

    # Print output shape and sample values
    print(f"\nOutput shape: {logits.shape} (batch_size, sequence_length, vocab_size)")
    print(f"Output data type: {logits.dtype}")

    # Print sample logits (first 5 positions of the first sequence)
    print("\nSample logits (first sequence, first 5 positions, first 5 values):")
    for position in range(min(5, L)):
        print(f"  Position {position}: {logits[0, position, :5]}")

    # Get predictions (token with highest logit at each position)
    predictions = jnp.argmax(logits, axis=-1)
    print("\nPredicted token IDs (first sequence, first 10 positions):")
    print(predictions[0, :10])

    # Test the predict function
    print("\nTesting predict function...")
    # Use a shorter
    short_seq = x_BxL[:, :10]
    print(f"Input sequence shape: {short_seq.shape}")

    # Predict 5 tokens
    k = 5
    original, predicted = model.apply(params, short_seq, k, method=model.predict)

    # Get predictions (token with highest logit at each position)
    predictions = jnp.argmax(logits, axis=-1)
    print("\nPredicted token IDs (first sequence, first 10 positions):")
    print(predictions[0, :10])

    print("\nDone!")


if __name__ == "__main__":
    main()
