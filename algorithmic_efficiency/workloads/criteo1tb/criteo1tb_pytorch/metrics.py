import torch
import torch.nn.functional as F
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



def roc_auc_score(logits, targets):
    """Pytorch implementation almost follows sklearn
    Args:
        targets (Tensor):
        logits (Tensor):
    """
    device = targets.device
    targets.squeeze_()
    logits.squeeze_()
    if targets.shape != logits.shape:
        raise TypeError(F"Shapre of targets and logits must match. Got {targets.shape()} and {logits.shape()}.")

    desc_score_indices = torch.argsort(logits, descending=True)
    logits = logits[desc_score_indices]
    targets = targets[desc_score_indices]

    distinct_value_indices = torch.nonzero(logits[1:] - logits[:-1], as_tuple=False).squeeze()
    threshold_idxs = torch.cat([distinct_value_indices, torch.tensor([targets.numel() - 1], device=device)])

    tps = torch.cumsum(targets, dim=0)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = torch.cat([torch.zeros(1, device=device), tps])
    fps = torch.cat([torch.zeros(1, device=device), fps])

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    area = torch.trapz(tpr, fpr)

    return area




if __name__=="__main__":
    batch_size = 141000
    logits = torch.randn(batch_size, 1, device='cuda')
    targets = torch.randint(low=0, high=2, size=(batch_size, 1), device='cuda')
    print(logits)
    print(targets)
    print(per_example_sigmoid_binary_cross_entropy(logits, targets))
    #print(log_loss(logits.long(), targets))

    #targets.sum()
