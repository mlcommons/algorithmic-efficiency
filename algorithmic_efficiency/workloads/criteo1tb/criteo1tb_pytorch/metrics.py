"""Binary sigmoid cross-entropy, AUCROC, and mAP metrics."""

import torch


def per_example_sigmoid_binary_cross_entropy(logits, targets, mask_batch):
  ls = torch.nn.LogSigmoid()
  log_p = ls(logits)
  log_not_p = ls(-logits)
  per_example_losses = -1.0 * (targets * log_p + (1 - targets) * log_not_p)
  if mask_batch is not None:
    mask_batch = torch.reshape(mask_batch, (mask_batch.shape[0],))
    per_example_losses *= mask_batch
  return per_example_losses
