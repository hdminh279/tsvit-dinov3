# TSViT-DINOv3: Factorized Spatio-Temporal Vision Transformer for Crop Segmentation

Multi-temporal crop classification and dense semantic segmentation on Sentinel-2 L2A satellite image time series (SITS) with official French RPG 2022 ground truth.

Inherited and improved from **TSViT** ([Tarasiou et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Tarasiou_ViTs_for_SITS_Vision_Transformers_for_Satellite_Image_Time_Series_CVPR_2023_paper.html)) and [MichailTarasiou/DeepSatModels](https://github.com/MichailTarasiou/DeepSatModels).

---

## Architecture Overview

TSViT separates time-series satellite reasoning into factorized temporal and spatial stages:

1. **Spectral Adapter:** Projects 10 Sentinel-2 bands (10m/20m resampled to 10m) into patch tokens ($24 \times 24 \to 144$ tokens with $2 \times 2$ patch size).
2. **Temporal Transformer:** Attends across timestamps $T$ using Day-of-Year (DOY) positional encodings and per-class learnable query tokens.
3. **Spatial Transformer Backbone (Upgraded to DINO-v3):** Replaces the randomly initialized vanilla ViT spatial encoder with self-supervised DINO-v3 representations. Sharpens crop parcel boundaries, reduces edge bleeding, and prevents overfitting on small geographic regions.
4. **Dense Segmentation Head:** Projects patch representations directly to pixel-level logits `[B, num_classes, 24, 24]`.

| Original TSViT Architecture | Improved TSViT-DINOv3 Architecture |
| :---: | :---: |
| ![Original TSViT](image/architecture.png) | ![TSViT-DINOv3](image/architecture_dinov3.png) |

---

## Loss Function: TSViT-DINOv3 vs. Original Paper

#### Original Paper (CVPR 2023)

The original implementation in `DeepSatModels` uses manual masking with an auxiliary binary mask $M \in \{0, 1\}^{B \times H \times W}$:

$$\mathcal{L}_{\text{orig}} = - \frac{1}{\sum_{i} M_i} \sum_{i: M_i = 1} \log \frac{\exp(z_{i, y_i})}{\sum_{c=1}^C \exp(z_{i, c})}$$

- Required `torch.gather()` on flattened logits indexed by boolean expanded masks:

```python
# DeepSatModels/metrics/loss_functions.py
masked_logits_flat = logits_flat[mask_flat.repeat(1, nclasses)].view(-1, nclasses)
masked_target_flat = target_flat[mask_flat].unsqueeze(dim=-1).to(torch.int64)
masked_log_probs_flat = torch.nn.functional.log_softmax(masked_logits_flat)
masked_losses_flat = -torch.gather(masked_log_probs_flat, dim=1, index=masked_target_flat)
loss = masked_losses_flat.mean()
```

#### TSViT-DINOv3

Leverages PyTorch native fused operator with `ignore_index`:

$$\mathcal{L}_{\text{DINOv3}} = - \frac{1}{\sum_{i: y_i \neq 0} 1} \sum_{i: y_i \neq 0} \log \frac{\exp(z_{i, y_i})}{\sum_{c=0}^{C-1} \exp(z_{i, c})}$$

- Direct execution on model outputs:

```python
# train.py
criterion = nn.CrossEntropyLoss(ignore_index=0)
loss = criterion(logits, y)  # logits: [B, C, H, W], y: [B, H, W]
```

---

## References

```bibtex
@inproceedings{tarasiou2023vits,
  title={ViTs for SITS: Vision Transformers for Satellite Image Time Series},
  author={Tarasiou, Michail and Chavez, Erik and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={10418--10428},
  year={2023}
}
```
