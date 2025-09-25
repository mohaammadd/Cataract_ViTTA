import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.metrics import accuracy
from utils.checkpoint import save_checkpoint

def train_model(model, train_set, config, logger):
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    device = config["device"]
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss, total_acc = 0, 0

        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            logits, _ = model(videos)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy(logits, labels)

        avg_loss, avg_acc = total_loss / len(train_loader), total_acc / len(train_loader)
        logger.logger.info(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")

        # TensorBoard logging
        logger.log_scalar("train/loss", avg_loss, epoch+1)
        logger.log_scalar("train/acc", avg_acc, epoch+1)

        # Save checkpoint
        is_best = avg_acc > best_acc
        if is_best:
            best_acc = avg_acc

        save_checkpoint({
            "epoch": epoch+1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "acc": avg_acc,
            "config": config
        }, is_best, config["save_dir"])

    return model
