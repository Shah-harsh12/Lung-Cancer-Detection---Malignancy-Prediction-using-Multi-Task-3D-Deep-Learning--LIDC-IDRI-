# training/losses.py

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """
    Combined loss for:
        - Detection (binary classification)
        - Malignancy (regression)
    """

    def __init__(
        self,
        lambda_det=1.0,
        lambda_mal=0.5,
        pos_weight=None,  # optional for imbalance
    ):
        super().__init__()

        if pos_weight is not None:
            self.det_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.det_loss = nn.BCEWithLogitsLoss()

        self.mal_loss = nn.SmoothL1Loss()

        self.lambda_det = lambda_det
        self.lambda_mal = lambda_mal

    def forward(self, det_logits, mal_pred, det_target, mal_target):

        det_target = det_target.float().unsqueeze(1)
        mal_target = mal_target.float().unsqueeze(1)

        loss_det = self.det_loss(det_logits, det_target)
        loss_mal = self.mal_loss(mal_pred, mal_target)

        total_loss = (
            self.lambda_det * loss_det +
            self.lambda_mal * loss_mal
        )

        return total_loss, loss_det, loss_mal
