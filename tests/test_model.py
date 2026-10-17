"""
Unit tests for TSViT-DINOv3 architecture and model export utilities:
- Forward pass tensor shapes: [B, num_classes, 24, 24]
- Backward pass & gradient flow
- Metrics computation
- ONNX and PyTorch checkpoint export
"""

import os
import unittest
import torch

from src.models.spectral_adapter import SpectralAdapter
from src.models.temporal_transformer import TemporalTransformer
from src.models.dinov3_backbone import DINOv3SpatialEncoder
from src.models.tsvit_dinov3 import TSViT_DINOv3
from src.utils.metrics import compute_segmentation_metrics
from src.utils.export import save_checkpoint, export_onnx


class TestModelArchitecture(unittest.TestCase):
    def setUp(self):
        self.B = 2
        self.T = 8
        self.C = 10
        self.H = 24
        self.W = 24
        self.num_classes = 19
        self.dim = 64  # Compact dim for fast test execution
        self.test_dir = "/tmp/test_tsvit_model"
        os.makedirs(self.test_dir, exist_ok=True)

    def test_spectral_adapter(self):
        """Verify Spectral Adapter maps [B, T, 10, 24, 24] -> [B, T, 144, dim]."""
        adapter = SpectralAdapter(in_channels=10, dim=self.dim, patch_size=2, adapter_type="conv")
        x = torch.randn(self.B, self.T, 10, self.H, self.W)
        tokens = adapter(x)
        self.assertEqual(tokens.shape, (self.B, self.T, 144, self.dim))

    def test_temporal_transformer(self):
        """Verify Temporal Transformer returns per-class queries [B, 144, num_classes, dim]."""
        temp_transf = TemporalTransformer(
            dim=self.dim,
            num_classes=self.num_classes,
            depth=2,
            heads=2,
            dim_head=32
        )
        tokens = torch.randn(self.B, self.T, 144, self.dim)
        doy = torch.linspace(0.0, 1.0, self.T).unsqueeze(0).expand(self.B, -1)
        cls_tokens = temp_transf(tokens, doy)
        self.assertEqual(cls_tokens.shape, (self.B, 144, self.num_classes, self.dim))

    def test_dinov3_spatial_encoder(self):
        """Verify DINOv3 spatial encoder preserves spatial token dimensions."""
        encoder = DINOv3SpatialEncoder(dim=self.dim, depth=2, heads=2, dim_head=32)
        spatial_in = torch.randn(self.B * self.num_classes, 144, self.dim)
        spatial_out = encoder(spatial_in)
        self.assertEqual(spatial_out.shape, (self.B * self.num_classes, 144, self.dim))

    def test_full_tsvit_dinov3_forward_and_backward(self):
        """Verify end-to-end forward pass yields [B, 19, 24, 24] and backward pass propagates gradients."""
        model = TSViT_DINOv3(
            img_res=24,
            patch_size=2,
            in_channels=10,
            num_classes=self.num_classes,
            dim=self.dim,
            temporal_depth=2,
            spatial_depth=2,
            heads=2,
            dim_head=32
        )

        x = torch.randn(self.B, self.T, 10, 24, 24, requires_grad=True)
        doy = torch.linspace(0.0, 1.0, self.T).unsqueeze(0).expand(self.B, -1)

        logits = model(x, doy)
        self.assertEqual(logits.shape, (self.B, self.num_classes, 24, 24))

        # Test backward pass
        targets = torch.randint(0, self.num_classes, (self.B, 24, 24))
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_metrics_computation(self):
        """Verify segmentation metrics (OA, F1, mIoU)."""
        preds = torch.tensor([[1, 2], [2, 1]])
        targets = torch.tensor([[1, 2], [2, 1]])
        metrics = compute_segmentation_metrics(preds, targets, num_classes=self.num_classes)
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['macro_f1'], 1.0)

    def test_checkpoint_and_onnx_export(self):
        """Verify checkpoint saving and ONNX format export."""
        model = TSViT_DINOv3(
            img_res=24,
            patch_size=2,
            in_channels=10,
            num_classes=self.num_classes,
            dim=self.dim,
            temporal_depth=1,
            spatial_depth=1,
            heads=2,
            dim_head=32
        )

        pt_path = os.path.join(self.test_dir, "test_model.pt")
        onnx_path = os.path.join(self.test_dir, "test_model.onnx")

        save_checkpoint(model, pt_path, epoch=1, metrics={'f1': 0.85})
        self.assertTrue(os.path.exists(pt_path))

        export_onnx(model, onnx_path, num_timesteps=self.T, img_res=24)
        self.assertTrue(os.path.exists(onnx_path))


if __name__ == "__main__":
    unittest.main()
