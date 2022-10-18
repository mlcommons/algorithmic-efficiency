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
