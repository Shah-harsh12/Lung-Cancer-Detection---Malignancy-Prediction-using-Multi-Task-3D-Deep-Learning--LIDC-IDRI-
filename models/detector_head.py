# models/detector_head.py

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """
    Binary classification head.
    Input:  (B, feature_dim)
    Output: (B, 1)  -> raw logits
    """

    def __init__(self, feature_dim=256, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        logits = self.classifier(features)
        return logits
