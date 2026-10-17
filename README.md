# Cải tiến TSViT của tác giả Michail Tarasiou áp dụng mô hình DINO-v3

## Tổng quan

Phân loại và phân đoạn ngữ nghĩa cây trồng nông nghiệp từ chuỗi thời gian vệ tinh Sentinel-2 L2A (10 kênh phổ, độ phân giải 10m/20m) kết hợp nhãn mặt đất RPG 2022.

Mô hình gốc kế thừa: **TSViT** (CVPR 2023).

```bibtex
@inproceedings{tarasiou2023vits,
  title={ViTs for SITS: Vision Transformers for Satellite Image Time Series},
  author={Tarasiou, Michail and Chavez, Erik and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={10418--10428},
  year={2023}
}
```

Mã nguồn: [MichailTarasiou/DeepSatModels](https://github.com/MichailTarasiou/DeepSatModels).

---

## Kiến trúc gốc của bài báo

![Kiến trúc gốc TSViT](image/architecture.png)

---

## Kiến trúc cải tiến với DINO-v3

![Kiến trúc cải tiến TSViT với DINO-v3](image/architecture_dinov3.png)

---

## Tóm tắt cải tiến

- **Vị trí thay thế:** Khối `Spatial Encoder` trong luồng xử lý phân tách không gian - thời gian của TSViT.
- **Lý do thay thế:** `Spatial Encoder` gốc dùng vanilla ViT khởi tạo ngẫu nhiên, học từ đầu trên tập ảnh vệ tinh nhỏ nên nhận diện ranh giới giữa bờ và thửa yếu và dễ quá khớp.
- **Mục đích:** Khai thác đặc trưng thị giác tự giám sát từ Foundation Model DINO-v3 để bắt biên thửa đất sắc nét, giữ nguyên `Temporal Encoder` qua chuỗi thời gian.

---

## Thay đổi cốt lõi trong mã nguồn

Mã nguồn tại `src/models/tsvit_dinov3.py`:

```python
self.spectral_adapter = SpectralAdapter(
    in_channels=in_channels,
    dim=dim,
    patch_size=patch_size
)

self.temporal_transformer = TemporalTransformer(
    dim=dim,
    num_classes=num_classes,
    depth=temporal_depth,
    heads=heads,
    dim_head=dim_head,
    mlp_ratio=mlp_ratio,
    dropout=dropout
)

self.spatial_backbone = DINOv3SpatialEncoder(
    dim=dim,
    depth=spatial_depth,
    heads=heads,
    dim_head=dim_head,
    mlp_ratio=mlp_ratio,
    dropout=dropout,
    model_name=dinov3_model_name
)

self.seg_head = nn.Sequential(
    nn.LayerNorm(dim),
    nn.Linear(dim, patch_size * patch_size)
)
```

---

## Đang phát triển

Đánh giá định lượng mức độ cải thiện độ chính xác (OA, Macro F1, mIoU) và hiệu năng suy luận so với mô hình gốc TSViT đang được tiến hành trên tập kiểm thử phân đoạn cây trồng RPG 2022.
