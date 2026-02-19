# evaluation/metrics.py

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)
from scipy.stats import pearsonr


def evaluate_model(model, dataloader, device="cuda"):

    model.eval()

    all_det_targets = []
    all_det_probs = []

    all_mal_targets = []
    all_mal_preds = []

    with torch.no_grad():
        for cubes, det_labels, mal_labels in dataloader:

            cubes = cubes.to(device)

            det_logits, mal_pred = model(cubes)

            det_probs = torch.sigmoid(det_logits).cpu().numpy().flatten()
            mal_pred = mal_pred.cpu().numpy().flatten()

            all_det_probs.extend(det_probs)
            all_det_targets.extend(det_labels.numpy())

            all_mal_preds.extend(mal_pred)
            all_mal_targets.extend(mal_labels.numpy())

    # -------------------------
    # Detection Metrics
    # -------------------------
    det_auc = roc_auc_score(all_det_targets, all_det_probs)

    det_preds = (np.array(all_det_probs) > 0.5).astype(int)

    det_acc = accuracy_score(all_det_targets, det_preds)
    det_precision = precision_score(all_det_targets, det_preds)
    det_recall = recall_score(all_det_targets, det_preds)
    det_f1 = f1_score(all_det_targets, det_preds)

    det_cm = confusion_matrix(all_det_targets, det_preds)

    # -------------------------
    # Malignancy Metrics
    # -------------------------
    mal_mae = mean_absolute_error(all_mal_targets, all_mal_preds)
    mal_rmse = np.sqrt(mean_squared_error(all_mal_targets, all_mal_preds))
    mal_corr, _ = pearsonr(all_mal_targets, all_mal_preds)

    results = {
        "Detection AUC": det_auc,
        "Detection Accuracy": det_acc,
        "Detection Precision": det_precision,
        "Detection Recall": det_recall,
        "Detection F1": det_f1,
        "Confusion Matrix": det_cm,
        "Malignancy MAE": mal_mae,
        "Malignancy RMSE": mal_rmse,
        "Malignancy Correlation": mal_corr,
    }

    return results
