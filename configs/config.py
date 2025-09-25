import os

CONFIG = {
    "exp_name": "swin_vitta_cataract",
    "device": "cuda",
    "seed": 42,

    # Data
    "train_videos": "./data/centerA/videos",
    "train_ann": "./data/centerA/annotations.csv",
    "test_videos": "./data/centerB/videos",
    "test_ann": "./data/centerB/annotations.csv",
    "num_frames": 32,
    "img_size": 224,

    # Training
    "batch_size": 2,
    "num_epochs": 20,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "save_dir": "./checkpoints",

    # ViTTA
    "tta_lr": 1e-5,
    "tta_lambda": 0.1,
    "tta_alpha": 0.1,
}
