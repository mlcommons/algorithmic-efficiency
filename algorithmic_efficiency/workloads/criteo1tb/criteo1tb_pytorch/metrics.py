import numpy as np
import torch
import sklearn.metrics

torch.manual_seed(0)



def per_example_sigmoid_binary_cross_entropy(logits, targets):
    """Computes the sigmoid binary cross entropy per example.
    Args:
      logits: float array of shape (batch, output_shape).
      targets: float array of shape (batch, output_shape).
      weights: None or float array of shape (batch,).
    Returns:
      Sigmoid binary cross entropy computed per example, shape (batch,).
    """
    ls = torch.nn.LogSigmoid()
    log_p = ls(logits)
    log_not_p = ls(-logits)
    per_example_losses = -1.0 * (targets * log_p + (1 - targets) * log_not_p)
    per_example_losses = (per_example_losses).reshape(per_example_losses.shape[0], -1)
    return torch.sum(per_example_losses, dim=1)






if __name__=="__main__":
    batch_size = 141000
    logits = torch.randn(batch_size, 1)
    targets = torch.randint(low=0, high=2, size=(batch_size, 1))
    print(logits)
    print(targets)
    print(per_example_sigmoid_binary_cross_entropy(logits, targets))

    targets.sum()
