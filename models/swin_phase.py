import torch
import torch.nn as nn
import timm


class VideoSwinPhaseClassifier(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str = "swin_tiny_patch4_window7_224"):
        """
        Swin Transformer backbone for video phase classification.

        Args:
            num_classes (int): number of surgical phases.
            backbone_name (str): timm model name, e.g.
                                 - "swin_tiny_patch4_window7_224" (~28M params)
                                 - "swin_small_patch4_window7_224" (~50M params)
                                 - "swin_base_patch4_window7_224" (~88M params)
        """
        super().__init__()

        # Load Swin backbone from timm
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0
        )
        hidden_dim = self.backbone.num_features

        # Classification head
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): video batch of shape (B, T, C, H, W)
                              B = batch size
                              T = number of frames (e.g., 16)
                              C = channels (3)
                              H, W = frame size (224x224)

        Returns:
            logits (torch.Tensor): (B, num_classes)
            feats (torch.Tensor): (B, hidden_dim), pooled video features
        """
        B, T, C, H, W = x.shape

        # Flatten temporal dimension → treat frames as images
        x = x.view(B * T, C, H, W)

        # Extract features per frame
        feats = self.backbone(x)  # (B*T, hidden_dim)

        # Reshape back into (B, T, hidden_dim)
        feats = feats.view(B, T, -1)

        # Temporal average pooling
        feats = feats.mean(1)  # (B, hidden_dim)

        # Classification
        logits = self.fc(feats)  # (B, num_classes)

        return logits, feats
