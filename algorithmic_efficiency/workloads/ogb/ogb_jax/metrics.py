# Forked from Flax example which can be found here:
# https://github.com/google/flax/blob/main/examples/ogbg_molpcba/train.py

import numpy as np
import jax
import jax.numpy as jnp
import flax
from clu import metrics
from sklearn.metrics import average_precision_score


def predictions_match_labels(*, logits: jnp.ndarray, labels: jnp.ndarray,
                             **kwargs) -> jnp.ndarray:
  """Returns a binary array indicating where predictions match the labels."""
  del kwargs  # Unused.
  preds = (logits > 0)
  return (preds == labels).astype(jnp.float32)


@flax.struct.dataclass
class MeanAveragePrecision(
    metrics.CollectingMetric.from_outputs(('logits', 'labels', 'mask'))):
  """Computes the mean average precision (mAP) over different tasks."""

  def compute(self):
    # Matches the official OGB evaluation scheme for mean average precision.
    labels = self.values['labels']
    logits = self.values['logits']
    mask = self.values['mask']

    mask = mask.astype(np.bool)

    probs = jax.nn.sigmoid(logits)
    num_tasks = labels.shape[1]
    average_precisions = np.full(num_tasks, np.nan)

    # Note that this code is slow (~1 minute).
    for task in range(num_tasks):
      # AP is only defined when there is at least one negative data
      # and at least one positive data.
      if np.sum(labels[:, task] == 0) > 0 and np.sum(labels[:, task] == 1) > 0:
        is_labeled = mask[:, task]
        average_precisions[task] = average_precision_score(
            labels[is_labeled, task], probs[is_labeled, task])

    # When all APs are NaNs, return NaN. This avoids raising a RuntimeWarning.
    if np.isnan(average_precisions).all():
      return np.nan
    return np.nanmean(average_precisions)


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  accuracy: metrics.Average.from_fun(predictions_match_labels)
  loss: metrics.Average.from_output('loss')
  mean_average_precision: MeanAveragePrecision

