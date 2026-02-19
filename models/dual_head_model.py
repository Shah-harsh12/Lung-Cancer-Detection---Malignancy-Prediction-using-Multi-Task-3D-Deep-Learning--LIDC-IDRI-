# models/dual_head_model.py

import torch
import torch.nn as nn

from models.backbone3d import Backbone3D
from models.detector_head import DetectionHead
from models.malignancy_head import MalignancyHead


class DualHeadModel(nn.Module):
    """
    Multi-task lung model:
        - Detection (binary classification)
        - Malignancy (regression)

    Input:
        (B, 1, 64, 64, 64)

    Outputs:
        detection_logits: (B, 1)
        malignancy_score: (B, 1)
    """

    def __init__(self, feature_dim=256):
        super().__init__()

        self.backbone = Backbone3D(feature_dim=feature_dim)

        self.detector_head = DetectionHead(feature_dim=feature_dim)
        self.malignancy_head = MalignancyHead(feature_dim=feature_dim)

    def forward(self, x):

        features = self.backbone(x)

        detection_logits = self.detector_head(features)
        malignancy_score = self.malignancy_head(features)

        return detection_logits, malignancy_score
