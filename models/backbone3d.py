# models/backbone3d.py

import torch
import torch.nn as nn


# ---------------------------------
# Basic 3D Residual Block
# ---------------------------------
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride,
                    bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ---------------------------------
# 3D Backbone
# ---------------------------------
class Backbone3D(nn.Module):
    """
    Input:  (B, 1, 64, 64, 64)
    Output: (B, feature_dim)
    """

    def __init__(self, feature_dim=256):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResidualBlock3D(32, 64, stride=2)   # 32³
        self.layer2 = ResidualBlock3D(64, 128, stride=2)  # 16³
        self.layer3 = ResidualBlock3D(128, 256, stride=2) # 8³

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x):

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.global_pool(x)  # (B, 256, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)

        features = self.fc(x)  # (B, feature_dim)

        return features
