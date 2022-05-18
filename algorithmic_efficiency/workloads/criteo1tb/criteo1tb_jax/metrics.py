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
  elif weights.shape == targets.shape[:weights.ndim]:
    # Add extra dimension if weights.shape is a prefix of targets.shape
    # so that multiplication can be broadcasted.
    weights = jnp.expand_dims(
        weights, axis=tuple(range(weights.ndim, targets.ndim)))
  elif weights.shape != targets.shape:
    raise ValueError('Incorrect shapes. Got shape %s weights and %s targets.' %
                    (str(weights.shape), str(targets.shape)))
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
  per_example_losses = (per_example_losses).reshape(
      per_example_losses.shape[0], -1)
  return jnp.sum(per_example_losses, axis=-1)


def _binary_auc_shape_fix_check(x, shape_error_msg):
  """Assert that the input shape is compatible, or fix it."""
  # One-hot targets, assumed to be shape [N, 2].
  if len(x.shape) == 2:
    if x.shape[1] > 2:
      raise ValueError(shape_error_msg)
    if x.shape[1] == 1:
      x = np.squeeze(x, axis=1)
    elif x.shape[1] == 2:
      # Binary AUC wants the labels/probabilities for the positive class, which
      # is the second element in the (n, 2) shaped array.
      x = x[:, 1]
  elif len(x.shape) > 2:
    raise ValueError(shape_error_msg)
  return x


def _binary_auc_shape_fix(targets, logits, weights, metric_name):
  """Ensure shapes are valid and convert them to dense shapes for sklearn.

  If inputs are shape (n, 2), we slice out the second column via x[:, 1]. If the
  inputs are shape (n, 1), we np.squeeze only the second dimension away. If the
  inputs are shape (n.), they are left untouched. If they are any other shape
  then a ValueError is raised.

  Args:
    targets: np.array of target labels, of shape (n,) or (n, 2).
    logits: np.array of model logits, of shape (n,) or (n, 2).
    weights: np.array of example weights, of shape (n,) or (n, 2).
    metric_name: the name of the metrics being checked, used for error messages.
  Returns:
    A triple of (targets, logits, weights) that now all have shape (n,).
  """
  shape_error_msg = (
      f'Inputs for {metric_name} should be of shape (n,) or (n, 2). Received '
      f'targets={targets.shape}, logits={logits.shape}, '
      f'weights={weights.shape}.')
  targets = _binary_auc_shape_fix_check(targets, shape_error_msg)
  logits = _binary_auc_shape_fix_check(logits, shape_error_msg)
  weights = _binary_auc_shape_fix_check(weights, shape_error_msg)
  # This happens if weights are None
  if np.all(np.isnan(weights)):
    weights = None
  # We need weights to be the exact same shape as targets, not just
  # compatible for broadcasting, so multiply by ones of the right shape.
  weights = np.ones(targets.shape) * _conform_weights_to_targets(
      weights, targets)
  return targets, logits, weights


@flax.struct.dataclass
class BinaryMeanAveragePrecision(
    metrics.CollectingMetric.from_outputs(('logits', 'targets', 'mask'))):
  """Computes the mean average precision for a binary classifier on CPU."""

  def compute(self):
    values = super().compute()
    # Ensure the arrays are numpy and not jax.numpy.
    values = {k: np.array(v) for k, v in values.items()}
    targets, logits, weights = _binary_auc_shape_fix(
        values['targets'],
        values['logits'],
        values['mask'],
        'BinaryMeanAveragePrecision')

    valid_targets = targets[weights > 0]
    targets_sum = np.sum(valid_targets)
    # Do not compute AUC if positives only have one class.
    if targets_sum == 0 or targets_sum == len(valid_targets):
      return 0.0
    probs = expit(logits)  # Sigmoid.
    return sklearn.metrics.average_precision_score(
        targets, probs, sample_weight=weights)


@flax.struct.dataclass
class BinaryAUCROC(
    metrics.CollectingMetric.from_outputs(('targets', 'logits', 'mask'))):
  """Compute the AUC-ROC for binary classification on the CPU."""

  def compute(self):
    values = super().compute()
    # Ensure the arrays are numpy and not jax.numpy.
    values = {k: np.array(v) for k, v in values.items()}
    targets, logits, weights = _binary_auc_shape_fix(
        values['targets'],
        values['logits'],
        values['mask'],
        'BinaryAUCROC')
    valid_targets = targets[weights > 0]
    targets_sum = np.sum(valid_targets)
    # Do not compute AUC if all labels are the same.
    if targets_sum == 0 or targets_sum == len(valid_targets):
      return 0.0
    positive_probs = expit(logits)  # Sigmoid.
    return sklearn.metrics.roc_auc_score(
        targets, positive_probs, sample_weight=weights)
