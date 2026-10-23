"""
Loss functions for Multi-Temporal Satellite Image Time Series (SITS) Segmentation.
Inherits MaskedCrossEntropyLoss from DeepSatModels, handling valid pixel masking
and excluding the void class (label 19 in PASTIS24).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class MaskedCrossEntropyLoss(nn.Module):
    """
    Masked Cross Entropy Loss for satellite semantic segmentation.
    Excludes void pixels (e.g., label 19 in PASTIS24) so that they contribute
    neither to the loss nor to the gradients.

    Supports:
      - logits: [B, C, H, W] (channel-first) or [B, H, W, C] (channel-last)
      - ground_truth: target tensor [B, H, W] or (target, mask)
    """
    def __init__(self, num_classes: int = 19, ignore_index: int = 19, reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        ground_truth: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        if isinstance(ground_truth, (list, tuple)):
            if len(ground_truth) == 2:
                target, mask = ground_truth
            else:
                target = ground_truth[0]
                mask = None
        else:
            target = ground_truth
            mask = None

        # Standardize logits to [B, H, W, C]
        if logits.dim() == 4:
            if logits.shape[1] == self.num_classes:
                # [B, C, H, W] -> [B, H, W, C]
                logits = logits.permute(0, 2, 3, 1)

        # Standardize target to [B, H, W]
        if target.dim() == 4 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        elif target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        c = logits.shape[-1]
        logits_flat = logits.reshape(-1, c)
        target_flat = target.reshape(-1).long()

        if mask is not None:
            if mask.dim() == 4 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            elif mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            valid_mask = mask.reshape(-1).bool()
        else:
            valid_mask = (target_flat != self.ignore_index)

        z_valid = logits_flat[valid_mask]
        y_valid = target_flat[valid_mask]

        if y_valid.numel() == 0:
            if self.reduction == "mean":
                raise ValueError("No valid target pixels for mean reduction in MaskedCrossEntropyLoss.")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        log_probs = F.log_softmax(z_valid, dim=-1)
        loss = -log_probs.gather(1, y_valid.unsqueeze(1)).squeeze(1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
