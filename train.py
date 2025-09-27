import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.metrics import accuracy
from utils.checkpoint import save_checkpoint

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, total_acc, total_samples = 0, 0, 0

    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            logits, _ = model(videos)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_acc += accuracy(logits, labels) * labels.size(0)
            total_samples += labels.size(0)

    return total_loss / total_samples, total_acc / total_samples


def train_model(model, train_set, val_set, config, logger):
    train_loader = DataLoader(train_set, batch_size=config["batch_size"],
                              shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_set, batch_size=config["batch_size"],
                            shuffle=False, num_workers=config["num_workers"])

    device = config["device"]
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss, total_acc, total_samples = 0, 0, 0

        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            logits, _ = model(videos)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_acc += accuracy(logits, labels) * labels.size(0)
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.logger.info(
            f"Epoch [{epoch+1}/{config['num_epochs']}], "
            f"Train Loss={avg_loss:.4f}, Train Acc={avg_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

        logger.log_scalar("train/loss", avg_loss, epoch+1)
        logger.log_scalar("train/acc", avg_acc, epoch+1)
        logger.log_scalar("val/loss", val_loss, epoch+1)
        logger.log_scalar("val/acc", val_acc, epoch+1)

        # Save checkpoint based on validation accuracy
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        save_checkpoint({
            "epoch": epoch+1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "acc": val_acc,
            "config": config
        }, is_best, config["save_dir"])

    return model
