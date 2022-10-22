"""Binary sigmoid cross-entropy metric."""

import jax
import jax.numpy as jnp


def per_example_sigmoid_binary_cross_entropy(logits, targets, mask_batch):
  """Computes the sigmoid binary cross entropy per example.

  Args:
    logits: float array of shape (batch, output_shape).
    targets: float array of shape (batch, output_shape).
  Returns:
    Sigmoid binary cross entropy computed per example, shape (batch,).
  """
  log_p = jax.nn.log_sigmoid(logits)
  log_not_p = jax.nn.log_sigmoid(-logits)
  losses = -1.0 * (targets * log_p + (1 - targets) * log_not_p)
  if mask_batch is not None:
      mask_batch = jnp.reshape(mask_batch, (mask_batch.shape[0],))
      losses *= mask_batch
  return losses
