import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_scalars_from_tb(tb_logdir, tag):
    """
    Load scalar values from TensorBoard event files.

    Args:
        tb_logdir (str): directory containing TensorBoard logs
        tag (str): scalar tag (e.g., 'train/loss')

    Returns:
        steps (list[int]), values (list[float])
    """
    ea = EventAccumulator(tb_logdir)
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        raise ValueError(f"Tag {tag} not found in {tb_logdir}")

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def plot_scalar(tb_logdir, tag, title=None, save_path=None):
    """
    Plot a scalar curve from TensorBoard logs.

    Args:
        tb_logdir (str): log directory
        tag (str): scalar tag (e.g., 'train/acc')
        title (str): plot title
        save_path (str): optional save path
    """
    steps, values = load_scalars_from_tb(tb_logdir, tag)

    plt.figure(figsize=(7, 5))
    plt.plot(steps, values, marker="o", label=tag)
    plt.xlabel("Step")
    plt.ylabel(tag.split("/")[-1].capitalize())
    plt.title(title if title else tag)
    plt.grid(True)
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_multiple(tb_logdir, tags, title="Comparison", save_path=None):
    """
    Plot multiple scalar curves on one figure.

    Args:
        tb_logdir (str): log directory
        tags (list[str]): list of scalar tags
        title (str): plot title
        save_path (str): optional save path
    """
    plt.figure(figsize=(7, 5))

    for tag in tags:
        steps, values = load_scalars_from_tb(tb_logdir, tag)
        plt.plot(steps, values, label=tag)

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
