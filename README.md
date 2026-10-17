# TSViT-DINOv3: Multi-Temporal Satellite Crop Classification

Hệ thống phân loại và lập bản đồ cây trồng nông nghiệp từ chuỗi ảnh vệ tinh Sentinel-2 đa thời gian (time-series). Kiến trúc kế thừa cơ chế phân rã không gian - thời gian (Factorized Spatio-Temporal Attention) từ `DeepSatModels/`, tái cấu trúc khối Spatial Attention với backbone tự giám sát **DINO-v3**, chuẩn hóa dải phổ và nhãn thửa đất RPG 2022.

---

## 1. So sánh Kiến trúc & Các Điểm Thay Thế Kế Thừa từ `DeepSatModels/`

| Thành phần | `DeepSatModels/` (Nguyên bản) | `crop_classification_tsvit_dinov3/` (Mới) | Vị trí file mã nguồn can thiệp |
| :--- | :--- | :--- | :--- |
| **Spatial Backbone** | `space_transformer` (Standard ViT blocks ngẫu nhiên, không có trọng số tự giám sát) | **DINO-v3 Spatial Encoder**: Tích hợp biểu diễn thị giác tự giám sát SOTA (DINO-v3 / DINOv2) | `src/models/dinov3_backbone.py` (`DINOv3SpatialEncoder`) |
| **Spectral Adapter** | `Rearrange` + `nn.Linear` đơn thuần cho patch 2x2 | **SpectralAdapter**: Module chiếu 10 dải phổ vệ tinh ($10 \to D$) với GroupNorm & GELU | `src/models/spectral_adapter.py` (`SpectralAdapter`) |
| **Temporal Transformer** | Khối Temporal Attention trong `TSViTdense.py` | Tách module độc lập, vector hóa mã hóa Day of Year (DOY) tương thích ONNX tracing | `src/models/temporal_transformer.py` (`TemporalTransformer`) |
| **Kích thước Patch** | Phụ thuộc tập dữ liệu ($24 \times 24$ hoặc $48 \times 48$) | Cắt đồng bộ chuẩn hóa $24 \times 24$, đầu ra $[B, 19, 24, 24]$ | `src/data/transforms.py` (`SynchronousPatchCrop`) |
| **Pipeline dữ liệu** | Đọc các tệp `.pkl` có sẵn từ đĩa | Ingest tự động từ Google Earth Engine / Planetary Computer, lọc mây SCL, rasterize nhãn RPG 2022 | `src/data/ingest_gee.py`, `src/data/rpg_processor.py` |
| **Xuất mô hình** | Lưu PyTorch `.tar` checkpoint | Hỗ trợ xuất đồng thời `.pt` và `.onnx` tối ưu hóa cho web runtime | `src/utils/export.py` (`save_checkpoint`, `export_onnx`) |

---

## 2. Thông số Tiền xử lý Chuẩn từ `DeepSatModels/`

### 10 Dải phổ Sentinel-2 (Bands Indexing):
1. **B02** (Blue - 490 nm - 10m)
2. **B03** (Green - 560 nm - 10m)
3. **B04** (Red - 665 nm - 10m)
4. **B05** (Red Edge 1 - 705 nm - 20m $\to$ resample 10m)
5. **B06** (Red Edge 2 - 740 nm - 20m $\to$ resample 10m)
6. **B07** (Red Edge 3 - 783 nm - 20m $\to$ resample 10m)
7. **B08** (NIR broad - 842 nm - 10m)
8. **B8A** (NIR narrow - 865 nm - 20m $\to$ resample 10m)
9. **B11** (SWIR 1 - 1610 nm - 20m $\to$ resample 10m)
10. **B12** (SWIR 2 - 2190 nm - 20m $\to$ resample 10m)

### Tham số chuẩn hóa $\mu$ và $\sigma$ (Kế thừa từ PASTIS24 Fold-1):
- **Mean $\mu$**:
  `[1165.94, 1375.65, 1429.22, 1764.80, 2719.27, 3063.61, 3205.90, 3319.11, 2422.90, 1639.37]`
- **Std $\sigma$**:
  `[1942.62, 1881.92, 1959.38, 1867.22, 1754.59, 1769.40, 1784.86, 1767.71, 1458.96, 1299.28]`
- **DOY Scaling**: $\text{DOY} / 365.0001$.

---

## 3. Cấu trúc Dự án

```
crop_classification_tsvit_dinov3/
├── configs/
│   └── model_config.yaml          # Hyperparameters & Solver config
├── src/
│   ├── data/
│   │   ├── constants.py           # Band indices, mean/std, RPG 2022 mapping
│   │   ├── transforms.py          # SCL cloud masking, normalization, patch crop
│   │   ├── ingest_gee.py          # Google Earth Engine S2 ingestion
│   │   ├── ingest_planetary.py    # Microsoft Planetary Computer fallback
│   │   ├── rpg_processor.py       # RPG polygon processing & rasterization
│   │   ├── dataset_builder.py     # End-to-end dataset generation
│   │   ├── dataloader.py          # PyTorch Dataset & DataLoader
│   │   └── upload_kaggle.py       # Kaggle dataset-metadata & API upload
│   ├── models/
│   │   ├── spectral_adapter.py    # 10-band spectral projection layer
│   │   ├── dinov3_backbone.py     # DINO-v3 spatial encoder
│   │   ├── temporal_transformer.py# TSViT temporal self-attention
│   │   └── tsvit_dinov3.py        # Complete TSViT-DINOv3 network
│   └── utils/
│       ├── metrics.py             # OA, Macro F1, mIoU computation
│       └── export.py              # Export .pt and .onnx models
├── notebooks/
│   └── train_colab.ipynb          # Automated training notebook for Colab
├── app/
│   └── app.py                     # Interactive Streamlit Web Application
├── tests/
│   ├── test_data_pipeline.py      # Unit tests for Step 1
│   └── test_model.py              # Unit tests for Step 2
├── Dockerfile                     # Deployment container config
└── requirements.txt               # Dependencies
```

---

## 4. Hướng dẫn Chạy & Huấn luyện

### Khởi tạo môi trường:
```bash
uv venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Chạy Unit Tests:
```bash
python -m unittest tests/test_data_pipeline.py
python -m unittest tests/test_model.py
```

### Chạy Giao diện Trực quan:
```bash
streamlit run app/app.py
```
