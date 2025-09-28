CONFIG = {
    "exp_name": "swin_vitta_cataract",
    "seed": 42,
    "save_dir": "./checkpoints",

    # Dataset
    "train_root": "/content/drive/MyDrive/Phase_Dataset/Cat101/data",
    "test_root": "/content/drive/MyDrive/Phase_Dataset/Cat150/data",
    "num_frames": 16,
    "img_size": 224,
    "batch_size": 2,       # keep small for Colab GPU
    "num_workers": 0,      # avoid worker crash in Colab

    # Training
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_epochs": 10,
    "device": "cuda",

    # Backbone (Swin version)
    "backbone": "swin_tiny_patch4_window7_224",

    # Test-time Adaptation
    "tta_lr": 1e-5,
    "tta_lambda": 0.1,
    "tta_alpha": 0.1,
}
