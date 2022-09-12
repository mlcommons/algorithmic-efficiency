"""Binary sigmoid cross-entropy, AUCROC, and mAP metrics."""

from clu import metrics
import flax
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit
import sklearn.metrics


def _conform_weights_to_targets(weights, targets):
    """Conforms the shape of weights to targets to apply masking.

    We allow shape of weights to be a prefix of the shape of targets, for example
    for targets of shape (n_batches, n_tasks) we allow weights with shape
    (n_batches, n_tasks) or (n_batches, ). Add the necessary trailing dimensions
    of size 1 so that weights can be applied as a mask by a simple multiplication,
    (n_batches, 1) in this case.

    Args:
      weights: None or a numpy array which shape is a prefix of targets shape
      targets: numpy array to conform the weights to
    Returns:
      weights with proper dimensions added to apply it as a mask.
    """
    if weights is None:
        weights = jnp.ones_like(targets)
    elif weights.shape == targets.shape[: weights.ndim]:
        # Add extra dimension if weights.shape is a prefix of targets.shape
        # so that multiplication can be broadcasted.
        weights = jnp.expand_dims(
            weights, axis=tuple(range(weights.ndim, targets.ndim))
        )
    elif weights.shape != targets.shape:
        raise ValueError(
            "Incorrect shapes. Got shape %s weights and %s targets."
            % (str(weights.shape), str(targets.shape))
        )
    return weights


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
    per_example_losses = (per_example_losses).reshape(per_example_losses.shape[0], -1)
    return jnp.sum(per_example_losses, axis=-1)
