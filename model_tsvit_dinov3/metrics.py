"""
Evaluation metrics for multi-temporal satellite crop segmentation.
Computes Overall Accuracy (OA), Macro F1, Mean IoU (mIoU), and per-class metrics.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, jaccard_score


def compute_segmentation_metrics(
    preds: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    num_classes: int = 19,
    ignore_index: Optional[int] = None
) -> Dict[str, float]:
    """
    Args:
        preds: Predicted class indices [B, H, W] or [N]
        targets: Ground truth class indices [B, H, W] or [N]
        num_classes: Total number of target crop classes
        ignore_index: Optional class ID to mask out from metrics (e.g. 0 for background)
    Returns:
        Dict of evaluation metrics: accuracy, macro_f1, mean_iou, precision, recall
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

    if ignore_index is not None:
        valid_mask = targets_flat != ignore_index
        preds_flat = preds_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]

    if len(targets_flat) == 0:
        return {'accuracy': 0.0, 'macro_f1': 0.0, 'mean_iou': 0.0}

    acc = float(accuracy_score(targets_flat, preds_flat))
    f1 = float(f1_score(targets_flat, preds_flat, average='macro', zero_division=0))
    prec = float(precision_score(targets_flat, preds_flat, average='macro', zero_division=0))
    rec = float(recall_score(targets_flat, preds_flat, average='macro', zero_division=0))
    iou = float(jaccard_score(targets_flat, preds_flat, average='macro', zero_division=0))

    return {
        'accuracy': acc,
        'macro_f1': f1,
        'precision': prec,
        'recall': rec,
        'mean_iou': iou
    }
