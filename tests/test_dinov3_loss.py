"""
Unit tests for MaskedCrossEntropyLoss and DINOv3 spatial embedding.
Verifies valid-pixel cross entropy, void-pixel masking (label 19), and gradient flow.
"""

import unittest
import torch
import torch.nn as nn
from src.models.losses import MaskedCrossEntropyLoss
from src.models.dinov3_backbone import DINOv3PatchEmbedding


class TestDINOv3AndLoss(unittest.TestCase):
    def test_masked_cross_entropy_void_label(self):
        loss_fn = MaskedCrossEntropyLoss(ignore_index=19, reduction="mean")
        
        # Logits: [B, C, H, W] = [1, 19, 2, 2]
        logits = torch.randn(1, 19, 2, 2, requires_grad=True)
        # Target: 3 valid pixels (0, 1, 2) and 1 void pixel (19)
        target = torch.tensor([[[[0, 1], [2, 19]]]], dtype=torch.long)
        
        loss = loss_fn(logits, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        
        loss.backward()
        # Verify gradient at void pixel location is zero
        void_grad = logits.grad[0, :, 1, 1]
        self.assertTrue(torch.allclose(void_grad, torch.zeros_like(void_grad)))
        
        # Verify gradient at valid pixel is non-zero
        valid_grad = logits.grad[0, :, 0, 0]
        self.assertFalse(torch.allclose(valid_grad, torch.zeros_like(valid_grad)))

    def test_dinov3_patch_embedding_shape_and_gradients(self):
        # [B, T, C, H, W] = [1, 3, 10, 24, 24]
        x = torch.randn(1, 3, 10, 24, 24)
        embedder = DINOv3PatchEmbedding(dim=128, frame_chunk_size=1)
        
        out = embedder(x)
        # Output shape should be [(B * 144), T, dim] = [144, 3, 128]
        self.assertEqual(out.shape, (144, 3, 128))
        
        # Projection should be trainable
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(embedder.projection.weight.grad)
        self.assertFalse(torch.allclose(embedder.projection.weight.grad, torch.zeros_like(embedder.projection.weight.grad)))


if __name__ == '__main__':
    unittest.main()
