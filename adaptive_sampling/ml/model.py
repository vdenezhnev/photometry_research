"""ResNet-based classifier for frame pairs."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class PairQualityModel(nn.Module):
    """Два кадра → общий ResNet18 → concat → бинарный класс (good/bad)."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        self._mobilenet = False

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = models.resnet18(weights=weights)
            feat_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
            self.head = nn.Sequential(
                nn.Linear(feat_dim * 2, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
            )
        elif backbone == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.mobilenet_v3_small(weights=weights)
            self.encoder = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
            feat_dim = 576
            self._mobilenet = True
            self.head = nn.Sequential(nn.Linear(feat_dim * 2, 128), nn.ReLU(), nn.Linear(128, 1))
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        fa = self.encoder(img_a)
        fb = self.encoder(img_b)
        x = torch.cat([fa, fb], dim=1)
        return self.head(x).squeeze(1)
