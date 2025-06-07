from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import lax, random

import flax.linen as nn
from flax.linen.module import Module, compact, merge_param
from flax.typing import PRNGKey


# Custom Layers
class Dropout(Module):
    """Create a dropout layer.
    Forked from https://flax-linen.readthedocs.io/en/latest/_modules/flax/linen/stochastic.html#Dropout.
    The reference dropout implementation is modified support changes to dropout rate during training by:
    1) adding rate argument to the __call__ method
    2) removing the if-else condition to check for edge cases, which will trigger a recompile for jitted code

    .. note::
      When using :meth:`Module.apply() <flax.linen.Module.apply>`, make sure
      to include an RNG seed named ``'dropout'``. Dropout isn't necessary for
      variable initialization.

    Example usage::

      >>> import flax.linen as nn
      >>> import jax, jax.numpy as jnp

      >>> class MLP(nn.Module):
      ...   @nn.compact
      ...   def __call__(self, x, train):
      ...     x = nn.Dense(4)(x)
      ...     x = nn.Dropout(0.5, deterministic=not train)(x)
      ...     return x

      >>> model = MLP()
      >>> x = jnp.ones((1, 3))
      >>> variables = model.init(jax.random.key(0), x, train=False) # don't use dropout
      >>> model.apply(variables, x, train=False) # don't use dropout
      Array([[-0.17875527,  1.6255447 , -1.2431065 , -0.02554005]], dtype=float32)
      >>> model.apply(variables, x, train=True, rngs={'dropout': jax.random.key(1)}) # use dropout
      Array([[-0.35751054,  3.2510893 ,  0.        ,  0.        ]], dtype=float32)

    Attributes:
      rate: the dropout probability.  (_not_ the keep rate!)
      broadcast_dims: dimensions that will share the same dropout mask
      deterministic: if false the inputs are scaled by ``1 / (1 - rate)`` and
        masked, whereas if true, no mask is applied and the inputs are returned as
        is.
      rng_collection: the rng collection name to use when requesting an rng key.
    """

    rate: float | None = None
    broadcast_dims: Sequence[int] = ()
    deterministic: bool | None = None
    rng_collection: str = "dropout"
    legacy: bool = True

    @compact
    def __call__(
        self,
        inputs,
        deterministic: bool | None = None,
        rate: float | None = None,
        rng: PRNGKey | None = None,
    ):
        """Applies a random dropout mask to the input.

        Args:
          inputs: the inputs that should be randomly masked.
          deterministic: if false the inputs are scaled by ``1 / (1 - rate)`` and
            masked, whereas if true, no mask is applied and the inputs are returned
            as is.
          rate: the dropout probability.  (_not_ the keep rate!)
          rng: an optional PRNGKey used as the random key, if not specified, one
            will be generated using ``make_rng`` with the ``rng_collection`` name.

        Returns:
          The masked inputs reweighted to preserve mean.
        """
        deterministic = merge_param("deterministic", self.deterministic, deterministic)

        # Override self.rate if rate is passed to __call__
        if not (self.rate is not None and rate is not None):
            rate = merge_param("rate", self.rate, rate)

        if self.legacy:
            if rate == 0.0:
                return inputs

            # Prevent gradient NaNs in 1.0 edge-case.
            if rate == 1.0:
                return jnp.zeros_like(inputs)

        if deterministic:
            return inputs

        keep_prob = 1.0 - rate
        if rng is None:
            rng = self.make_rng(self.rng_collection)
        broadcast_shape = list(inputs.shape)
        for dim in self.broadcast_dims:
            broadcast_shape[dim] = 1
        mask = random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
        mask = jnp.broadcast_to(mask, inputs.shape)
        return lax.select(mask, inputs, jnp.zeros_like(inputs))


# Utilities for debugging
def print_jax_model_summary(model, fake_inputs):
    """Prints a summary of the jax module."""
    tabulate_fn = nn.tabulate(
        model,
        jax.random.PRNGKey(0),
        console_kwargs={"force_terminal": False, "force_jupyter": False, "width": 240},
    )
    print(tabulate_fn(fake_inputs, train=False))
