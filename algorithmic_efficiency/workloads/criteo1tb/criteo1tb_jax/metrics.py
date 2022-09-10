"""Binary sigmoid cross-entropy metric."""

import jax
import jax.numpy as jnp


def per_example_sigmoid_binary_cross_entropy(logits, targets):
  """Computes the sigmoid binary cross entropy per example.

  Args:
    logits: float array of shape (batch, output_shape).
    targets: float array of shape (batch, output_shape).
    weights: None or float array of shape (batch,).
  Returns:
    Sigmoid binary cross entropy computed per example, shape (batch,).
  """
  log_p = jax.nn.log_sigmoid(logits)
  log_not_p = jax.nn.log_sigmoid(-logits)
  per_example_losses = -1.0 * (targets * log_p + (1 - targets) * log_not_p)
  per_example_losses = (per_example_losses).reshape(per_example_losses.shape[0],
                                                    -1)
  return jnp.sum(per_example_losses, axis=-1)
