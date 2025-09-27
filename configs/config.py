CONFIG = {
    # Experiment
    "exp_name": "swin_vitta_cataract",
    "seed": 42,
    "save_dir": "./checkpoints",

    # Dataset
    "train_root": "/content/drive/MyDrive/Phase_Dataset/Cat101/data",  # Center A
    "test_root": "/content/drive/MyDrive/Phase_Dataset/Cat150/data",   # Center B
    "num_frames": 16,
    "img_size": 224,
    "batch_size": 4,
    "num_workers": 4,

    # Training
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_epochs": 20,
    "device": "cuda",

    # Test-time Adaptation (ViTTA)
    "tta_lr": 1e-5,
    "tta_lambda": 0.1,   # weight for consistency loss
    "tta_alpha": 0.1,    # EMA momentum
}
