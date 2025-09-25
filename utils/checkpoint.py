import os
import torch
from typing import Dict, Any

def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    save_dir: str,
    filename: str = "checkpoint.pth"
) -> None:
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary to save (e.g., model state, optimizer state).
        is_best (bool): If True, also saves a copy as 'best_model.pth'.
        save_dir (str): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file (default: 'checkpoint.pth').
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)
    torch.save(state, ckpt_path)
    if is_best:
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(state, best_path)