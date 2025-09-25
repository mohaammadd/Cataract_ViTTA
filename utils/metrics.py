import torch
from typing import Any

def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes the accuracy for classification outputs.

    Args:
        outputs (torch.Tensor): Model outputs (logits or probabilities), shape (batch_size, num_classes).
        targets (torch.Tensor): Ground truth labels, shape (batch_size,).

    Returns:
        float: Accuracy value between 0 and 1.
    """
    preds = torch.argmax(outputs, dim=1)
    # Ensure targets are 1D and on the same device
    targets = targets.view(-1).to(preds.device)
    if preds.numel() == 0:
        return 0.0
    return (preds == targets).float().mean().item()