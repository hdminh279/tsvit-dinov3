"""
Unit tests for Step 1 data pipeline components:
- Normalization (S2 and DOY)
- RPG 2022 parcel rasterization
- Synchronous 24x24 patch cropping
- S2CropDatasetBuilder & DataLoader integration
"""

import os
import unittest
import numpy as np
import torch

from src.data.constants import S2_MEAN_10B, S2_STD_10B, CROP_PATCH_SIZE, NUM_CLASSES
from src.data.transforms import NormalizeS2, NormalizeDOY, SynchronousPatchCrop, CloudMaskSCL
from src.data.rpg_processor import RPGProcessor
from src.data.dataset_builder import S2CropDatasetBuilder
from src.data.dataloader import S2CropPatchDataset, get_dataloaders
from src.data.upload_kaggle import KaggleDatasetUploader


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.T = 12
        self.C = 10
        self.H = 48
        self.W = 48
        self.test_dir = "/tmp/test_s2_data_pipeline"
        os.makedirs(self.test_dir, exist_ok=True)

    def test_normalization_shapes_and_values(self):
        """Verify (x - mu) / sigma normalization on tensor and array."""
        normalizer = NormalizeS2()
        
        # Test on numpy array
        raw_arr = np.repeat(S2_MEAN_10B.reshape(1, 10, 1, 1), self.T, axis=0)
        raw_arr = np.repeat(np.repeat(raw_arr, self.H, axis=2), self.W, axis=3)
        norm_arr = normalizer(raw_arr)
        
        self.assertEqual(norm_arr.shape, (self.T, self.C, self.H, self.W))
        # Since input equals mean, normalized output must be zero
        np.testing.assert_allclose(norm_arr, 0.0, atol=1e-5)

        # Test on torch.Tensor
        tensor_in = torch.randn(2, self.T, 10, self.H, self.W)
        norm_tensor = normalizer(tensor_in)
        self.assertEqual(norm_tensor.shape, (2, self.T, 10, self.H, self.W))

    def test_doy_normalization(self):
        """Verify DOY / 365.0001 scaling."""
        doy_norm = NormalizeDOY()
        doys = np.array([1.0, 180.0, 365.0], dtype=np.float32)
        scaled = doy_norm(doys)
        self.assertTrue(np.all(scaled >= 0.0))
        self.assertTrue(np.all(scaled < 1.0))

    def test_rpg_rasterization(self):
        """Verify agricultural polygon rasterization to target grid."""
        processor = RPGProcessor()
        bounds = (2.0, 48.0, 2.1, 48.1)
        from shapely.geometry import box
        parcels = [
            (box(2.01, 48.01, 2.05, 48.05), '1'),
            (box(2.06, 48.06, 2.09, 48.09), '2')
        ]
        raster = processor.rasterize_parcels(parcels, bounds, (self.H, self.W))

        self.assertEqual(raster.shape, (self.H, self.W))
        self.assertTrue(np.all(raster >= 0))
        self.assertTrue(np.all(raster < NUM_CLASSES))

    def test_synchronous_24x24_patch_crop(self):
        """Verify synchronous patch crop yields exactly [T, 10, 24, 24] and [24, 24]."""
        cropper = SynchronousPatchCrop(patch_size=CROP_PATCH_SIZE, min_valid_ratio=0.0)
        images = np.random.randn(self.T, 10, self.H, self.W).astype(np.float32)
        doy = np.arange(self.T).astype(np.float32)
        label = np.random.randint(0, NUM_CLASSES, (self.H, self.W), dtype=np.int64)

        samples = cropper(images, doy, label)
        expected_patches = (self.H // CROP_PATCH_SIZE) * (self.W // CROP_PATCH_SIZE)
        self.assertEqual(len(samples), expected_patches)

        first_sample = samples[0]
        self.assertEqual(first_sample['image'].shape, (self.T, 10, 24, 24))
        self.assertEqual(first_sample['doy'].shape, (self.T,))
        self.assertEqual(first_sample['label'].shape, (24, 24))

    def test_dataset_builder_and_dataloader(self):
        """Verify end-to-end dataset builder and PyTorch DataLoader."""
        builder = S2CropDatasetBuilder(patch_size=24)
        dummy_images = np.ones((10, 10, 48, 48), dtype=np.float32) * 1500.0
        dummy_doy = np.linspace(10, 200, 10).astype(np.float32)
        dummy_labels = np.ones((48, 48), dtype=np.int64) * 2
        result = builder.build_dataset_from_arrays(
            images_raw=dummy_images,
            doy_raw=dummy_doy,
            label_raster=dummy_labels,
            output_dir=self.test_dir,
            num_samples_target=4,
            save_format="pt"
        )
        self.assertEqual(result['total_samples'], 4)

        # Verify DataLoader
        train_loader, val_loader = get_dataloaders(self.test_dir, batch_size=2, val_split=0.5)
        for batch in train_loader:
            self.assertEqual(batch['inputs'].shape, (2, 10, 10, 24, 24))
            self.assertEqual(batch['doy'].shape, (2, 10))
            self.assertEqual(batch['label'].shape, (2, 24, 24))
            break

    def test_kaggle_metadata_generation(self):
        """Verify dataset-metadata.json generation."""
        uploader = KaggleDatasetUploader(dataset_dir=self.test_dir, username="test_researcher")
        meta_path = uploader.create_metadata()
        self.assertTrue(os.path.exists(meta_path))


if __name__ == "__main__":
    unittest.main()
