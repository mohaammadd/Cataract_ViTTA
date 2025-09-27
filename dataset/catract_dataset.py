import os
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

class CataractVideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, img_size=224, train=True):
        """
        Args:
            root_dir (str): path to dataset (e.g., Phase_Dataset/Cat101/data)
            num_frames (int): number of frames per clip
            img_size (int): resize dimension
            train (bool): apply augmentations if True
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.train = train

        # Collect all sub-videos or frame folders
        self.samples = []
        self.phase_to_idx = {}
        phase_id = 0

        for case in sorted(os.listdir(root_dir)):
            case_path = os.path.join(root_dir, case)
            if not os.path.isdir(case_path):
                continue

            for phase in sorted(os.listdir(case_path)):
                phase_path = os.path.join(case_path, phase)
                if not os.path.isdir(phase_path):
                    continue

                if phase not in self.phase_to_idx:
                    self.phase_to_idx[phase] = phase_id
                    phase_id += 1

                for item in os.listdir(phase_path):
                    item_path = os.path.join(phase_path, item)
                    if item.lower().endswith((".mp4", ".avi", ".mov")) or os.path.isdir(item_path) or item.lower().endswith((".jpg", ".png")):
                        self.samples.append((item_path, self.phase_to_idx[phase]))

        print(f"Found {len(self.samples)} samples across {len(self.phase_to_idx)} phases.")

        # Transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.augment = T.RandomApply(torch.nn.ModuleList([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.GaussianBlur(3)], p=0.3),
        ]), p=0.7) if train else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        if os.path.isfile(path) and path.lower().endswith((".mp4", ".avi", ".mov")):
            frames = self._load_video(path)
        elif os.path.isdir(path):  # folder of frames
            frames = self._load_frames_from_dir(path)
        elif path.lower().endswith((".jpg", ".png")):  # single image
            frames = [cv2.imread(path)[:, :, ::-1]]
        else:
            raise ValueError(f"Unsupported file type: {path}")

        # Sample num_frames frames
        if len(frames) >= self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
            frames = [frames[i] for i in indices]
        else:
            # Pad by repeating last frame
            frames += [frames[-1]] * (self.num_frames - len(frames))

        frames = [self.transform(f) for f in frames]
        if self.augment:
            frames = [self.augment(f) for f in frames]

        clip = torch.stack(frames, dim=0)  # (T, C, H, W)
        return clip, label

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def _load_frames_from_dir(self, path):
        frames = []
        for fname in sorted(os.listdir(path)):
            if fname.lower().endswith((".jpg", ".png")):
                img = cv2.imread(os.path.join(path, fname))[:, :, ::-1]
                frames.append(img)
        return frames