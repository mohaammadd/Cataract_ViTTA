import torch
import torch.nn as nn
from timm.models.swin_transformer_v2 import swin_transformer_v2_base

# If using timm >= 0.9, Video Swin is available as "swin_base_patch244_window877_kinetics400"
# but we can build a wrapper like this for clarity.

class VideoSwinPhaseClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Load Video Swin Transformer (Swin-B variant pretrained on Kinetics-400)
        self.backbone = torch.hub.load(
            'facebookresearch/pytorchvideo',
            'swin_base_patch244_window877_kinetics400',
            pretrained=pretrained
        )
        
        # Remove original classifier
        feat_dim = self.backbone.head.proj.in_features
        self.backbone.head.proj = nn.Identity()

        # New classifier for cataract phases
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        """
        x: (B, T, C, H, W) → permute to (B, C, T, H, W) for Video Swin
        """
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        feats = self.backbone(x)   # (B, D)
        logits = self.classifier(feats)

        return logits, feats
