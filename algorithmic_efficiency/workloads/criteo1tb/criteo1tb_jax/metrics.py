from clu import metrics
import flax
import jax
import jax.numpy as jnp


def per_example_sigmoid_binary_cross_entropy(logits, targets):
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
  return jnp.sum(losses.reshape(losses.shape[0], -1), axis=-1)


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  loss: metrics.Average.from_output('loss')
