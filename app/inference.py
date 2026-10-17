"""
Inference engine for TSViT-DINOv3 crop classification.
Executes multi-temporal satellite segmentation and aggregates crop statistics and confidences.
"""

from typing import Dict, List, Optional, Tuple, Union
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.tsvit_dinov3 import TSViT_DINOv3
from src.data.constants import (
    CROP_CLASSES_VI, CROP_CLASSES_EN, NUM_CLASSES,
    S2_MEAN_10B, S2_STD_10B, DOY_DIVISOR
)
from src.data.transforms import NormalizeS2, NormalizeDOY


class CropInferenceEngine:
    """
    Handles preprocessing, model inference, and output statistics calculation.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Auto-detect checkpoint if not provided
        if not checkpoint_path:
            candidates = [
                os.path.join(os.path.dirname(__file__), "best.pth"),
                os.path.join(PROJECT_ROOT, "checkpoints", "best.pth"),
                os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
            ]
            for c in candidates:
                if os.path.exists(c):
                    checkpoint_path = c
                    break
        self.model = TSViT_DINOv3(
            img_res=24,
            patch_size=2,
            in_channels=10,
            num_classes=NUM_CLASSES,
            dim=192,
            temporal_depth=4,
            spatial_depth=4,
            heads=4,
            dim_head=48
        ).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
                self.model.load_state_dict(state_dict, strict=False)
                print(f"[Inference] Loaded model weights from: {checkpoint_path}")
            except Exception as e:
                print(f"[Inference] Warning loading checkpoint: {e}. Using initialized model weights.")

        self.model.eval()
        self.normalizer = NormalizeS2()
        self.doy_normalizer = NormalizeDOY()

    def run_inference(
        self,
        images_raw: np.ndarray,  # [T, 10, 24, 24] or [T, 10, H, W]
        doy_raw: np.ndarray       # [T]
    ) -> Dict[str, Union[np.ndarray, Dict, List]]:
        """
        Runs inference and computes crop distribution and confidence percentages.
        """
        # Ensure 24x24 spatial resolution
        T, C, H, W = images_raw.shape
        if H != 24 or W != 24:
            # Center crop or resize to 24x24
            top = max(0, (H - 24) // 2)
            left = max(0, (W - 24) // 2)
            images_raw = images_raw[:, :, top:top+24, left:left+24]

        # Normalize inputs
        images_norm = self.normalizer(images_raw)
        doy_norm = self.doy_normalizer(doy_raw)

        x_tensor = torch.from_numpy(images_norm).unsqueeze(0).float().to(self.device)  # [1, T, 10, 24, 24]
        doy_tensor = torch.from_numpy(doy_norm).unsqueeze(0).float().to(self.device)   # [1, T]

        with torch.no_grad():
            logits = self.model(x_tensor, doy_tensor)  # [1, num_classes, 24, 24]
            probs = F.softmax(logits, dim=1).squeeze(0)  # [num_classes, 24, 24]
            confidences, pred_map = torch.max(probs, dim=0)  # [24, 24], [24, 24]

        pred_map_np = pred_map.cpu().numpy()
        conf_map_np = confidences.cpu().numpy()
        probs_np = probs.cpu().numpy()

        # Compute crop breakdown percentages and average confidence
        total_pixels = 24 * 24
        unique_classes, counts = np.unique(pred_map_np, return_counts=True)

        crop_stats = []
        for cls_id, count in zip(unique_classes, counts):
            cls_mask = pred_map_np == cls_id
            avg_conf = float(np.mean(conf_map_np[cls_mask]) * 100.0)
            percentage = float((count / total_pixels) * 100.0)
            name_vi = CROP_CLASSES_VI.get(int(cls_id), f"Loại {cls_id}")
            name_en = CROP_CLASSES_EN.get(int(cls_id), f"Class {cls_id}")

            crop_stats.append({
                'class_id': int(cls_id),
                'name_vi': name_vi,
                'name_en': name_en,
                'percentage': round(percentage, 2),
                'confidence': round(avg_conf, 1),
                'pixel_count': int(count)
            })

        # Sort by percentage descending
        crop_stats.sort(key=lambda x: x['percentage'], reverse=True)

        return {
            'prediction_map': pred_map_np,
            'confidence_map': conf_map_np,
            'crop_stats': crop_stats,
            'rgb_preview': self._create_rgb_composite(images_raw)
        }

    @staticmethod
    def _create_rgb_composite(images_raw: np.ndarray) -> np.ndarray:
        """Creates normalized RGB preview (B04: Red, B03: Green, B02: Blue) for middle timestamp."""
        mid_t = images_raw.shape[0] // 2
        # Bands: 2=B04(Red), 1=B03(Green), 0=B02(Blue)
        rgb = images_raw[mid_t, [2, 1, 0], :, :].transpose(1, 2, 0)  # [24, 24, 3]
        # Percentile clip and normalize to 0-255
        p2, p98 = np.percentile(rgb, (2, 98))
        if p98 > p2:
            rgb_norm = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
        else:
            rgb_norm = np.clip(rgb / 3000.0, 0.0, 1.0)
        return (rgb_norm * 255).astype(np.uint8)
