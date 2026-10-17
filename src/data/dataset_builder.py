"""
End-to-End dataset builder for Sentinel-2 time series and RPG 2022 crop labels.
Coordinates ingestion, normalization, RPG polygon rasterization, 24x24 patch cropping,
and dataset serialization (.pt / .npy).
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

    def build_synthetic_dataset(
        self,
        num_timestamps: int = 30,
        height: int = 144,
        width: int = 144,
        num_samples_target: int = 36,
        output_dir: str = "data/processed",
        save_format: str = "pt"
    ) -> Dict[str, Union[int, str]]:
        """
        Generates and saves a realistic synthetic dataset for pipeline validation and unit testing.
        """
        os.makedirs(output_dir, exist_ok=True)
        rng = np.random.default_rng(42)

        # 1. Generate multi-temporal DOYs (e.g. from day 15 to 330)
        doy_raw = np.linspace(15, 335, num_timestamps).astype(np.float32)

        # 2. Generate synthetic S2 10-band reflectance
        # Base values centered around PASTIS24 mean
        base_reflectance = S2_MEAN_10B.reshape(1, 10, 1, 1)
        variability = rng.normal(0, 0.3 * S2_STD_10B.reshape(1, 10, 1, 1), size=(num_timestamps, 10, height, width))
        images_raw = np.clip(base_reflectance + variability, 0, 10000).astype(np.float32)

        # 3. Generate synthetic RPG 2022 parcel polygons and rasterize
        bounds = (2.0, 48.0, 2.05, 48.05)  # Sample AOI near Paris / Île-de-France
        parcels = RPGProcessor.generate_synthetic_aoi_parcels(bounds, grid_rows=6, grid_cols=6)
        label_raster = self.rpg_processor.rasterize_parcels(parcels, bounds, (height, width))

        # 4. Crop into 24x24 samples
        samples = self.process_and_crop(images_raw, doy_raw, label_raster)

        # 5. Save to disk
        metadata_records = []
        for idx, sample in enumerate(samples[:num_samples_target]):
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
                'num_timesteps': num_timestamps,
                'patch_h': self.patch_size,
                'patch_w': self.patch_size,
                'num_bands': 10
            })

        df_meta = pd.DataFrame(metadata_records)
        df_meta.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

        return {
            'total_samples': len(metadata_records),
            'output_dir': output_dir,
            'patch_shape': f"[{num_timestamps}, 10, {self.patch_size}, {self.patch_size}]",
            'label_shape': f"[{self.patch_size}, {self.patch_size}]"
        }
