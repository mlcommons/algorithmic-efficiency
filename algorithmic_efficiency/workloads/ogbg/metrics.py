# Forked from Flax example which can be found here:
# https://github.com/google/flax/blob/main/examples/ogbg_molpcba/train.py
import os
from typing import Any

from clu import metrics
import flax
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import average_precision_score
import torch
import torch.distributed as dist

USE_PYTORCH_DDP = 'LOCAL_RANK' in os.environ
RANK = int(os.environ['LOCAL_RANK']) if USE_PYTORCH_DDP else 0
DEVICE = torch.device(f'cuda:{RANK}' if torch.cuda.is_available() else 'cpu')
N_GPUS = torch.cuda.device_count()


def predictions_match_labels(*,
                             logits: jnp.ndarray,
                             labels: jnp.ndarray,
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
    values = super().compute()
    labels = values['labels']
    logits = values['logits']
    mask = values['mask']

    if USE_PYTORCH_DDP:
      # Sync labels, logits, and masks across devices.
      all_values = [labels, logits, mask]
      for idx, array in enumerate(all_values):
        tensor = torch.as_tensor(array, device=DEVICE)
        # Assumes that the tensors on all devices have the same shape.
        all_tensors = [torch.zeros_like(tensor) for _ in range(N_GPUS)]
        dist.all_gather(all_tensors, tensor)
        all_values[idx] = torch.cat(all_tensors).cpu().numpy()
      labels, logits, mask = all_values

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


class AverageDDP(metrics.Average):
  """Supports syncing metrics for PyTorch distributed data parallel (DDP)."""

  def compute(self) -> Any:
    if USE_PYTORCH_DDP:
      # Sync counts across devices.
      total_tensor = torch.as_tensor(np.asarray(self.total), device=DEVICE)
      count_tensor = torch.as_tensor(np.asarray(self.count), device=DEVICE)
      dist.all_reduce(total_tensor)
      dist.all_reduce(count_tensor)
      # Hacky way to avoid FrozenInstanceError
      # (https://docs.python.org/3/library/dataclasses.html#frozen-instances).
      object.__setattr__(self, 'total', total_tensor.cpu().numpy())
      object.__setattr__(self, 'count', count_tensor.cpu().numpy())
    return super().compute()


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  accuracy: AverageDDP.from_fun(predictions_match_labels)
  loss: AverageDDP.from_output('loss')
  mean_average_precision: MeanAveragePrecision
