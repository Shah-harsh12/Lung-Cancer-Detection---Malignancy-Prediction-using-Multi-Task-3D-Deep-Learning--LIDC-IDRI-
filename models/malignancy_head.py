# models/malignancy_head.py

import torch
import torch.nn as nn


class MalignancyHead(nn.Module):
    """
    Regression head for malignancy prediction.

    Input:  (B, feature_dim)
    Output: (B, 1)  -> continuous malignancy score
    """

    def __init__(self, feature_dim=256, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        score = self.regressor(features)
        return score
