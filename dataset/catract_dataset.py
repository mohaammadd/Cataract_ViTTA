import torch
from torch.utils.data import Dataset
import glob, os
import numpy as np
import decord
from torchvision import transforms as T
import random

# -------- Custom Augmentations -------- #

class SaltPepperNoise(object):
    def __init__(self, prob=0.01):
        self.prob = prob

    def __call__(self, tensor):
        # tensor: (C,H,W) in [0,1]
        noisy = tensor.clone()
        mask = torch.rand_like(noisy[0])  # (H,W)
        noisy[:, mask < self.prob/2] = 0.0
        noisy[:, mask > 1 - self.prob/2] = 1.0
        return noisy


class GaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)


# -------- Dataset -------- #

class CataractVideoDataset(Dataset):
    """
    Loads cataract sub-videos (already split by phases).
    Each sample returns: (T,C,H,W), label
    """
    def __init__(self, video_dir, annotation_file, num_frames=16, img_size=224, train=True):
        self.video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
        self.annotations = self._load_annotations(annotation_file)
        self.num_frames = num_frames
        self.train = train

        base_transform = [
            T.Resize((img_size, img_size)),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ]

        if train:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                SaltPepperNoise(prob=0.01),
                GaussianNoise(std=0.05),
                *base_transform
            ])
        else:
            self.transform = T.Compose(base_transform)

    def _load_annotations(self, annotation_file):
        ann = {}
        with open(annotation_file, "r") as f:
            for line in f:
                path, label = line.strip().split(",")
                ann[path] = int(label)
        return ann

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)

        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        frames = vr.get_batch(indices).asnumpy()  # (T,H,W,C)

        # Convert to tensor
        frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0

        # Apply transform per frame
        frames = torch.stack([self.transform(f) for f in frames])

        label = self.annotations[os.path.basename(video_path)]
        return frames, label
