import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import warnings

class CataractVideoDataset(Dataset):
    PHASE_MAP = {
        "AnteriorChamberFlushing": "FlushingAntibiotic",
        "TonifyingAntibiotics": "FlushingAntibiotic",
        "CapsulePolishing": "CortexRemoval",
        # All other phases map to themselves
        "ViscoelasticSuction": "ViscoelasticSuction",
        "Viscoelastic": "Viscoelastic",
        "Phacoemulsification": "Phacoemulsification",
        "LensImplantation": "LensImplantation",
        "IrrigationAspiration": "IrrigationAspiration",
        "Incision": "Incision",
        "Idle": "Idle",
        "Hydrodissection": "Hydrodissection",
        "CortexRemoval": "CortexRemoval",
        "Capsulorhexis": "Capsulorhexis",
    }

    def __init__(self, root_dir, num_frames=16, img_size=224, train=True):
        self.samples = []
        self.num_frames = num_frames
        self.train = train
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.augment = T.RandomHorizontalFlip(p=0.5) if train else None

        self.phase_to_idx = {}
        idx = 0
        for case in sorted(os.listdir(root_dir)):
            case_path = os.path.join(root_dir, case)
            if not os.path.isdir(case_path):
                continue
            for phase in sorted(os.listdir(case_path)):
                phase_path = os.path.join(case_path, phase)
                if not os.path.isdir(phase_path):
                    continue
                # Map phase name to unified label
                unified_phase = self.PHASE_MAP.get(phase, phase)
                if unified_phase not in self.phase_to_idx:
                    self.phase_to_idx[unified_phase] = idx
                    idx += 1
                for item in os.listdir(phase_path):
                    if item.lower().endswith((".mp4", ".avi", ".mov")):
                        self.samples.append((os.path.join(phase_path, item),
                                             self.phase_to_idx[unified_phase]))

    def __len__(self):
        return len(self.samples)

    def _load_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            warnings.warn(f"Could not open video {path}, skipping...")
            return None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames if len(frames) > 0 else None

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_video(video_path)

        if frames is None:  # fallback if broken
            return torch.zeros(self.num_frames, 3, 224, 224), torch.tensor(label)

        if len(frames) >= self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
            frames = [frames[i] for i in indices]
        else:
            frames += [frames[-1]] * (self.num_frames - len(frames))

        frames = [self.transform(Image.fromarray(f)) for f in frames]
        if self.augment:
            frames = [self.augment(f) for f in frames]
        clip = torch.stack(frames, dim=0)
        return clip, label
