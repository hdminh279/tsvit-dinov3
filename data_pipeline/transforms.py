"""
Data transformations and cloud masking for Sentinel-2 time-series.
Implements band normalization, 20m->10m spatial resampling, SCL cloud masking,
and synchronous 24x24 patch cropping.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from .constants import S2_MEAN_10B, S2_STD_10B, DOY_DIVISOR, CROP_PATCH_SIZE


class NormalizeS2:
    """Normalize 10-band Sentinel-2 time series with pasture/cropland benchmark statistics."""
    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = S2_MEAN_10B if mean is None else mean
        self.std = S2_STD_10B if std is None else std

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: Array or Tensor of shape [T, 10, H, W] or [10, H, W]
        Returns:
            Normalized tensor/array of same shape: (x - mean) / std
        """
        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            if x.ndim == 4:  # [T, 10, H, W]
                mean = mean.view(1, 10, 1, 1)
                std = std.view(1, 10, 1, 1)
            elif x.ndim == 3:  # [10, H, W]
                mean = mean.view(10, 1, 1)
                std = std.view(10, 1, 1)
            elif x.ndim == 5:  # [B, T, 10, H, W]
                mean = mean.view(1, 1, 10, 1, 1)
                std = std.view(1, 1, 10, 1, 1)
            return (x - mean) / std
        else:
            mean = self.mean
            std = self.std
            if x.ndim == 4:
                mean = mean.reshape(1, 10, 1, 1)
                std = std.reshape(1, 10, 1, 1)
            elif x.ndim == 3:
                mean = mean.reshape(10, 1, 1)
                std = std.reshape(10, 1, 1)
            return (x - mean) / std


class NormalizeDOY:
    """Scale Day of Year to range [0, 1)."""
    def __init__(self, divisor: float = DOY_DIVISOR):
        self.divisor = divisor

    def __call__(self, doy: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return doy / self.divisor


class CloudMaskSCL:
    """
    Cloud masking using Sentinel-2 L2A Scene Classification Layer (SCL).
    Classes in SCL:
      0: NO_DATA
      1: SATURATED_DEFECTIVE
      2: DARK_AREA_PIXELS
      3: CLOUD_SHADOWS (Masked)
      4: VEGETATION (Valid)
      5: NOT_VEGETATED (Valid)
      6: WATER (Valid)
      7: UNCLASSIFIED (Valid / Borderline)
      8: CLOUD_MEDIUM_PROBABILITY (Masked)
      9: CLOUD_HIGH_PROBABILITY (Masked)
      10: THIN_CIRRUS (Masked)
      11: SNOW (Masked)
    """
    def __init__(self, valid_classes: Tuple[int, ...] = (4, 5, 6, 7)):
        self.valid_classes = set(valid_classes)

    def get_mask(self, scl: np.ndarray) -> np.ndarray:
        """Returns boolean mask where True = valid pixel, False = cloudy/shadow."""
        valid_mask = np.isin(scl, list(self.valid_classes))
        return valid_mask

    def mask_images(self, x: np.ndarray, scl: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        """
        Args:
            x: Sentinel-2 image array [T, C, H, W]
            scl: SCL layer array [T, H, W]
        Returns:
            Cloud-masked image array
        """
        valid = self.get_mask(scl)[:, np.newaxis, :, :]  # [T, 1, H, W]
        x_clean = np.where(valid, x, fill_value)
        return x_clean


class ResampleBands:
    """
    Resample Sentinel-2 20m bands to match 10m grid resolution via bilinear interpolation.
    10m bands: B2, B3, B4, B8 (indices 0, 1, 2, 6)
    20m bands: B5, B6, B7, B8A, B11, B12 (indices 3, 4, 5, 7, 8, 9)
    """
    @staticmethod
    def resample_tensor(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [..., H, W]
            target_hw: Tuple of (target_height, target_width)
        Returns:
            Resampled tensor of shape [..., target_h, target_w]
        """
        orig_shape = x.shape
        h_orig, w_orig = orig_shape[-2], orig_shape[-1]
        if (h_orig, w_orig) == target_hw:
            return x
        # Reshape to (Batch, Channels, H, W) for F.interpolate
        flattened = x.view(-1, 1, h_orig, w_orig)
        resampled = F.interpolate(flattened, size=target_hw, mode='bilinear', align_corners=False)
        return resampled.view(*orig_shape[:-2], target_hw[0], target_hw[1])


class SynchronousPatchCrop:
    """
    Cuts multi-temporal imagery X [T, 10, H, W] and spatial label map Y [H, W]
    synchronously into square patches of size [patch_size, patch_size] (default 24x24).
    """
    def __init__(self, patch_size: int = CROP_PATCH_SIZE, stride: Optional[int] = None, min_valid_ratio: float = 0.2):
        self.patch_size = patch_size
        self.stride = patch_size if stride is None else stride
        self.min_valid_ratio = min_valid_ratio

    def __call__(
        self,
        images: np.ndarray,
        doy: np.ndarray,
        label: np.ndarray
    ) -> List[Dict[str, np.ndarray]]:
        """
        Args:
            images: [T, 10, H, W]
            doy: [T]
            label: [H, W]
        Returns:
            List of sample dicts containing:
                'image': [T, 10, patch_size, patch_size]
                'doy': [T]
                'label': [patch_size, patch_size]
        """
        T, C, H, W = images.shape
        samples = []

        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                patch_img = images[:, :, y:y + self.patch_size, x:x + self.patch_size]
                patch_label = label[y:y + self.patch_size, x:x + self.patch_size]

                # Filter out patches containing mostly invalid/empty labels if threshold is set
                valid_pixels = np.count_nonzero(patch_label > 0)
                total_pixels = self.patch_size * self.patch_size
                if self.min_valid_ratio > 0 and (valid_pixels / total_pixels) < self.min_valid_ratio:
                    continue

                samples.append({
                    'image': np.ascontiguousarray(patch_img, dtype=np.float32),
                    'doy': np.ascontiguousarray(doy, dtype=np.float32),
                    'label': np.ascontiguousarray(patch_label, dtype=np.int64)
                })

        return samples
