import torch
import json
import os
from torch.utils.data import random_split, DataLoader
from configs.default_config import CONFIG
from dataset.cataract_dataset import CataractVideoDataset
from models.swin_phase import VideoSwinPhaseClassifier
from utils.logger import ExperimentLogger
from utils.seed import set_seed
from train import train_model
from test_vitta import evaluate_with_vitta

if __name__ == "__main__":
    set_seed(CONFIG["seed"])

    # Setup experiment logger
    logger = ExperimentLogger(CONFIG["save_dir"], CONFIG["exp_name"], config=CONFIG)

    # Save config as JSON inside log directory
    config_path = os.path.join(logger.log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=4)
    logger.logger.info(f"Config saved to {config_path}")

    # Load training dataset (Center A)
    full_train = CataractVideoDataset(CONFIG["train_root"],
                                      num_frames=CONFIG["num_frames"],
                                      img_size=CONFIG["img_size"],
                                      train=True)

    # Split into train/val
    val_size = int(0.2 * len(full_train))
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])
    logger.logger.info(f"Train size: {train_size}, Val size: {val_size}")

    # Load test dataset (Center B)
    test_set = CataractVideoDataset(CONFIG["test_root"],
                                    num_frames=CONFIG["num_frames"],
                                    img_size=CONFIG["img_size"],
                                    train=False)

    num_classes = len(full_train.phase_to_idx)
    model = VideoSwinPhaseClassifier(num_classes)

    # Training + validation
    logger.logger.info("==== Training on Center A (with validation split) ====")
    model = train_model(model, train_set, val_set, CONFIG, logger)

    # Compute source stats (on training set only)
    logger.logger.info("==== Precomputing Source Stats (from train set) ====")
    model.eval()
    feats_all = []
    device = CONFIG["device"]
    with torch.no_grad():
        for videos, _ in DataLoader(train_set, batch_size=2):
            videos = videos.to(device)
            _, feats = model(videos)
            feats_all.append(feats)
    feats_all = torch.cat(feats_all, dim=0)
    mu_src, var_src = feats_all.mean(0), feats_all.var(0)
    source_stats = {"last": (mu_src, var_src)}

    # Test-time adaptation
    logger.logger.info("==== Test-time Adaptation on Center B ====")
    evaluate_with_vitta(model, test_set, source_stats, CONFIG, logger)

    # Close TensorBoard writer
    logger.close()
