import torch
from torch.utils.data import DataLoader
from models.vitta_adapter import ViTTAAdapter
from utils.metrics import accuracy

def evaluate_with_vitta(model, test_set, source_stats, config, logger):
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    adapter = ViTTAAdapter(model, source_stats,
                           lr=config["tta_lr"],
                           lambda_cons=config["tta_lambda"],
                           alpha=config["tta_alpha"],
                           device=config["device"])
    device = config["device"]
    model = model.to(device)

    correct, total = 0, 0

    for step, (videos, labels) in enumerate(test_loader, start=1):
        videos, labels = videos.to(device), labels.to(device)

        # Perform adaptation step
        losses = adapter.adapt_step(videos)

        # Log adaptation losses
        logger.log_scalar("tta/loss_total", losses["total"], step)
        logger.log_scalar("tta/loss_align", losses["align"], step)
        logger.log_scalar("tta/loss_cons", losses["cons"], step)

        # Inference after adaptation
        model.eval()
        with torch.no_grad():
            logits, _ = model(videos)
            correct += (torch.argmax(logits, 1) == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    logger.logger.info(f"Test-time Adaptation Accuracy = {acc:.4f}")
    logger.log_scalar("tta/accuracy", acc, step=0)
    return acc
