from flax import linen as nn
import jax.numpy as jnp

class LinearModel(nn.Module):
    vocab_size: int
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            512,
            kernel_init=nn.initializers.normal(0.02),
            bias_init=nn.initializers.zeros
        )(inputs)
        return nn.Dense(
            self.vocab_size,
            kernel_init=nn.initializers.normal(0.02),
            bias_init=nn.initializers.zeros
        )(x)
