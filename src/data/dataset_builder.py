"""
End-to-End dataset builder for Sentinel-2 time series and RPG 2022 crop labels.
Coordinates ingestion, normalization, RPG polygon rasterization, 24x24 patch cropping,
and dataset serialization (.pt / .npz).
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import json
import numpy as np
import torch
import pandas as pd

from .constants import S2_10BANDS, S2_MEAN_10B, S2_STD_10B, CROP_PATCH_SIZE, NUM_CLASSES
from .transforms import NormalizeS2, NormalizeDOY, SynchronousPatchCrop, CloudMaskSCL
from .rpg_processor import RPGProcessor


class S2CropDatasetBuilder:
    """
    Builds paired time-series satellite patches [T, 10, 24, 24] and crop labels [24, 24].
    """
    def __init__(
        self,
        patch_size: int = CROP_PATCH_SIZE,
        stride: int = CROP_PATCH_SIZE,
        min_valid_ratio: float = 0.2
    ):
        self.patch_size = patch_size
        self.normalizer = NormalizeS2()
        self.doy_normalizer = NormalizeDOY()
        self.cropper = SynchronousPatchCrop(patch_size=patch_size, stride=stride, min_valid_ratio=min_valid_ratio)
        self.rpg_processor = RPGProcessor()

    def process_and_crop(
        self,
        images_raw: np.ndarray,      # [T, 10, H, W] in Sentinel-2 DN / reflectance
        doy_raw: np.ndarray,         # [T] Day of Year (1 - 365)
        label_raster: np.ndarray,    # [H, W] integer crop classes
        scl_raster: Optional[np.ndarray] = None  # [T, H, W] optional cloud layer
    ) -> List[Dict[str, np.ndarray]]:
        """
        Runs normalization, cloud masking, and synchronous 24x24 patch cropping.
        """
        T, C, H, W = images_raw.shape
        assert C == 10, f"Expected 10 spectral channels, got {C}"
        assert label_raster.shape == (H, W), f"Label shape {label_raster.shape} must match spatial grid ({H}, {W})"

        # Optional SCL cloud masking
        if scl_raster is not None:
            masker = CloudMaskSCL()
            images_clean = masker.mask_images(images_raw, scl_raster)
        else:
            images_clean = images_raw

        # Normalize 10 bands: (x - mean) / std
        images_norm = self.normalizer(images_clean)

        # Scale DOY: doy / 365.0001
        doy_norm = self.doy_normalizer(doy_raw)

        # Extract 24x24 synchronous patches
        samples = self.cropper(images_norm, doy_norm, label_raster)
        return samples

    def build_dataset_from_arrays(
        self,
        images_raw: np.ndarray,
        doy_raw: np.ndarray,
        label_raster: np.ndarray,
        output_dir: str = "data/processed",
        scl_raster: Optional[np.ndarray] = None,
        num_samples_target: Optional[int] = None,
        save_format: str = "pt"
    ) -> Dict[str, Union[int, str]]:
        """
        Processes real multi-temporal satellite imagery and ground truth raster into 24x24 patches.
        """
        os.makedirs(output_dir, exist_ok=True)
        T, C, H, W = images_raw.shape

        samples = self.process_and_crop(images_raw, doy_raw, label_raster, scl_raster=scl_raster)
        if num_samples_target is not None:
            samples = samples[:num_samples_target]

        metadata_records = []
        for idx, sample in enumerate(samples):
            sample_id = f"patch_{idx:05d}"
            if save_format == "pt":
                file_path = os.path.join(output_dir, f"{sample_id}.pt")
                torch.save({
                    'image': torch.from_numpy(sample['image']).float(),
                    'doy': torch.from_numpy(sample['doy']).float(),
                    'label': torch.from_numpy(sample['label']).long()
                }, file_path)
            else:
                file_path = os.path.join(output_dir, f"{sample_id}.npz")
                np.savez_compressed(
                    file_path,
                    image=sample['image'],
                    doy=sample['doy'],
                    label=sample['label']
                )

            metadata_records.append({
                'sample_id': sample_id,
                'file_name': os.path.basename(file_path),
                'num_timesteps': T,
                'patch_h': self.patch_size,
                'patch_w': self.patch_size,
                'num_bands': 10
            })

        df_meta = pd.DataFrame(metadata_records)
        df_meta.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

        return {
            'total_samples': len(metadata_records),
            'output_dir': output_dir,
            'patch_shape': f"[{T}, 10, {self.patch_size}, {self.patch_size}]",
            'label_shape': f"[{self.patch_size}, {self.patch_size}]"
        }
