import torch
import json
import numpy as np

def save_source_stats(stats, path_json, path_pt):
    """
    Save source statistics (mean, var) to both JSON and .pt files.
    Args:
        stats (dict): {layer: (mu, var)} with torch tensors
        path_json (str): Path to save JSON file
        path_pt (str): Path to save .pt file
    """
    # Save as .pt (PyTorch native)
    torch.save(stats, path_pt)

    # Convert tensors to lists for JSON
    stats_json = {k: {'mean': v[0].tolist(), 'var': v[1].tolist()} for k, v in stats.items()}
    with open(path_json, 'w') as f:
        json.dump(stats_json, f, indent=4)

def load_source_stats(path_json=None, path_pt=None, device='cpu'):
    """
    Load source statistics from JSON or .pt file.
    Args:
        path_json (str): Path to JSON file
        path_pt (str): Path to .pt file
        device (str): Device to load tensors onto
    Returns:
        dict: {layer: (mu, var)} with torch tensors
    """
    if path_pt is not None:
        stats = torch.load(path_pt, map_location=device)
        return stats
    elif path_json is not None:
        with open(path_json, 'r') as f:
            stats_json = json.load(f)
        stats = {k: (torch.tensor(v['mean'], device=device), torch.tensor(v['var'], device=device))
                 for k, v in stats_json.items()}
        return stats
    else:
        raise ValueError('Provide either path_json or path_pt')
