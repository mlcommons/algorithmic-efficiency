"""Binary sigmoid cross-entropy, AUCROC, and mAP metrics."""

import torch


def per_example_sigmoid_binary_cross_entropy(logits, targets):
  ls = torch.nn.LogSigmoid()
  log_p = ls(logits)
  log_not_p = ls(-logits)
  per_example_losses = \
    -1.0 * (targets * log_p + (1 - targets) * log_not_p)
  per_example_losses = per_example_losses.reshape(per_example_losses.shape[0],
                                                  -1)
  return torch.sum(per_example_losses, dim=1)


def roc_auc_score(logits, targets):
  device = targets.device
  targets.squeeze_()
  logits.squeeze_()
  if targets.shape != logits.shape:
    raise TypeError(f'Shapre of targets and logits must match. '
                    f'Got {targets.shape()} and {logits.shape()}.')

  desc_score_indices = torch.argsort(logits, descending=True)
  logits = logits[desc_score_indices]
  targets = targets[desc_score_indices]

  distinct_value_indices = torch.nonzero(
      logits[1:] - logits[:-1], as_tuple=False).squeeze()
  threshold_idxs = torch.cat([
      distinct_value_indices,
      torch.tensor([targets.numel() - 1], device=device)
  ])

  tps = torch.cumsum(targets, dim=0)[threshold_idxs]
  fps = 1 + threshold_idxs - tps

  tps = torch.cat([torch.zeros(1, device=device), tps])
  fps = torch.cat([torch.zeros(1, device=device), fps])

  fpr = fps / fps[-1]
  tpr = tps / tps[-1]

  area = torch.trapz(tpr, fpr)

  return area
