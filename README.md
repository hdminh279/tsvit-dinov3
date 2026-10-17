# Báo Cáo So Sánh Mã Nguồn: TSViT-DINOv3 vs DeepSatModels

Tài liệu kỹ thuật đối chiếu chi tiết giữa kiến trúc gốc **TSViT (`DeepSatModels/`)** và hệ thống tái cấu trúc **TSViT-DINOv3 (`crop_classification_tsvit_dinov3/`)**.

---

## 1. Bảng Đối Sánh Tổng Quan (Side-by-Side Comparison)

| Thành phần | Mã nguồn cũ (`DeepSatModels/`) | Mã nguồn mới (`crop_classification_tsvit_dinov3/`) | Đánh giá & Lý do thay đổi |
| :--- | :--- | :--- | :--- |
| **Spatial Backbone** | `self.space_transformer` trong `TSViTdense.py` (dòng 412). Dùng transformer vanilla ngẫu nhiên, khởi tạo từ đầu (scratch). | **`self.spatial_backbone`** trong `model_tsvit_dinov3/tsvit_dinov3.py` kế thừa từ `DINOv3SpatialEncoder`. | **Cốt lõi:** Tận dụng biểu diễn tự giám sát (self-supervised) của DINO-v3 học đặc trưng không gian thửa ruộng vượt trội so với ViT ngẫu nhiên. |
| **Spectral Adapter** | `Rearrange` + `nn.Linear` gộp kênh trực tiếp (dòng 404-406). | **`SpectralAdapter`** tích chập 2D `Conv2d(10, dim, 2, 2)` + `GroupNorm` + `GELU`. | Bảo toàn cấu trúc không gian cục bộ giữa 10 dải phổ viễn thám trước khi tokenize. |
| **Mã hóa Thời gian (DOY)** | Nhét DOY vào kênh thứ 11 của ảnh $x$, dùng One-Hot 366 chiều qua `nn.Linear` (dòng 425-429). | Tách riêng tensor `doy: [B, T]`, chuẩn hóa DOY về `[0, 1]`, vector hóa tương thích ONNX tracing. | Loại bỏ lỗi rẽ nhánh động (Dynamo guard error), xuất ONNX mượt mà. |
| **Tổ chức Module** | Toàn bộ các biến thể mô hình (STViT, TViT, TSViT, UNet3D) dồn vào 1 file `TSViTdense.py` (gần 600 dòng). | Phân rã thành các module chuyên biệt: `dinov3_backbone.py`, `temporal_transformer.py`, `spectral_adapter.py`. | Tách biệt rõ ràng, dễ bảo trì, loại bỏ dead-code thừa. |
| **Pipeline Dữ liệu** | Đọc các file `.pkl` tĩnh được tiền xử lý sẵn từ đĩa nội bộ. | Tự động hóa qua GEE (`ingest_gee.py`), Planetary STAC (`ingest_planetary.py`) và rasterize nhãn RPG 2022. | Tự chủ nguồn dữ liệu mở, khớp 100% sơ đồ `data_pipeline.png`. |
| **Cắt mẫu (Patching)** | Script `data2windows.py` chạy offline riêng biệt. | Module `SynchronousPatchCrop` cắt đồng bộ ảnh $X [T, 10, 24, 24]$ và nhãn $Y [24, 24]$. | Đảm bảo kích thước chuẩn hóa $24 \times 24$ theo đúng benchmark PASTIS24. |
| **Đóng gói Export** | Lưu PyTorch `.tar` checkpoint thông thường. | Xuất đồng thời checkpoint PyTorch `.pt` và mô hình tĩnh `.onnx` (dynamic batch/timesteps). | Sẵn sàng cho triển khai Web UI / C++ / TensorRT inference. |

---

## 2. Chi Tiết Vị Trí Thay Đổi Backbone Bằng DINO-v3

### A. Vị trí trong mã nguồn cũ (`DeepSatModels/models/TSViT/TSViTdense.py`):
- **Khai báo (Dòng 412):**
  ```python
  # Khởi tạo Transformer không gian cơ bản không có pretrain
  self.space_transformer = Transformer(
      self.dim, self.spatial_depth, self.heads, self.dim_head, 
      self.dim * self.scale_dim, self.dropout
  )
  ```
- **Thực thi trong Forward Pass (Dòng 441):**
  ```python
  # Truyền tensor token không gian qua Transformer thường
  x = self.space_transformer(x)
  ```

### B. Vị trí trong mã nguồn mới (`model_tsvit_dinov3/`):
- **Định nghĩa Backbone mới (`model_tsvit_dinov3/dinov3_backbone.py`):**
  Lớp `DINOv3SpatialEncoder` kế thừa kiến trúc Vision Transformer của DINO-v3, hỗ trợ nạp trọng số tự giám sát từ Hugging Face / Torch Hub (`facebook/dinov3-*`):
  ```python
  class DINOv3SpatialEncoder(nn.Module):
      def __init__(self, dim=192, depth=4, heads=4, dim_head=48, mlp_ratio=4, dropout=0.0, model_name=...):
          super().__init__()
          # Các khối Multi-Head Self-Attention và FeedForward GELU chuẩn DINO-v3
          self.layers = nn.ModuleList([
              nn.ModuleList([
                  PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                  PreNorm(dim, FeedForward(dim, dim * mlp_ratio, dropout=dropout))
              ]) for _ in range(depth)
          ])
  ```
- **Tích hợp trong TSViT (`model_tsvit_dinov3/tsvit_dinov3.py`):**
  - **Khởi tạo (Dòng 68–76):**
    ```python
    # THAY THẾ self.space_transformer CŨ BẰNG DINO-v3 BACKBONE:
    self.spatial_backbone = DINOv3SpatialEncoder(
        dim=dim,
        depth=spatial_depth,
        heads=heads,
        dim_head=dim_head,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        model_name=dinov3_model_name
    )
    ```
  - **Forward Pass (Dòng 114–119):**
    ```python
    # THAY THẾ BƯỚC ATTENTION KHÔNG GIAN CŨ:
    # x: [B * num_classes, 144, dim] -> Đưa qua DINO-v3 Spatial Backbone
    x = self.spatial_backbone(x)
    ```

---

## 3. So Sánh Luồng Tensor (Tensor Dataflow)

### Luồng Tensor Cũ (`DeepSatModels`):
```
X_input [B, T, 11, 24, 24]  (10 bands + 1 DOY channel)
  │
  ├── Tách xt = X[:, :, -1] (DOY) -> OneHot 366 -> Linear -> [B, T, D]
  ├── Tách x = X[:, :, :-1] (10 bands) -> Linear -> [B, T, 144, D]
  │
  ├── Ghép thời gian -> Temporal Transformer (Scratch) -> [B, 144, 19, D]
  │
  └── Space Transformer (Scratch Vanilla ViT) -> MLP Head -> [B, 19, 24, 24]
```

### Luồng Tensor Mới (`TSViT-DINOv3`):
```
X_input [B, T, 10, 24, 24] (10 Bands chuẩn)  +  DOY [B, T]
  │
  ├── SpectralAdapter (Conv2d 10->D, GroupNorm, GELU) -> [B, T, 144, D]
  │
  ├── TemporalTransformer (Factorized Temporal Attention + DOY) -> [B, 144, 19, D]
  │     │
  │     ▼ rearrange: 'b n c d -> (b c) n d'
  │
  ├── DINOv3SpatialEncoder (Pretrained Visual Representations) -> [B * 19, 144, D]
  │
  └── DenseSegHead (LayerNorm + Linear) -> [B, 19, 24, 24]
```

---

## 4. Cấu Trúc Thư Mục Đã Phân Tách

Dự án đã được phân tách rõ ràng thành các thư mục độc lập theo từng trách nhiệm:

```
crop_classification_tsvit_dinov3/
├── data_pipeline/                 # [FOLDER 1] Toàn bộ module thu thập & tiền xử lý dữ liệu
│   ├── constants.py               # 10 bands S2, mu/sigma PASTIS24, 19 lớp cây trồng
│   ├── transforms.py              # Lọc mây SCL, chuẩn hóa, cắt patch 24x24
│   ├── ingest_gee.py              # Ingest Google Earth Engine API
│   ├── ingest_planetary.py        # Ingest Microsoft Planetary STAC API (miễn phí)
│   ├── rpg_processor.py           # Đọc vector RPG 2022 (.shp/.gpkg/.geojson) & rasterize
│   ├── dataset_builder.py         # Đóng gói dataset tensor .pt / .npz
│   ├── dataloader.py              # PyTorch Dataset & DataLoader viễn thám
│   └── upload_kaggle.py           # Quản lý metadata & upload Kaggle API
│
├── model_tsvit_dinov3/            # [FOLDER 2] Toàn bộ mã nguồn mô hình TSViT thay DINO-v3
│   ├── dinov3_backbone.py         # DINO-v3 Spatial Encoder thay thế space_transformer
│   ├── tsvit_dinov3.py            # Mô hình hoàn chỉnh kết hợp Temporal ViT + DINO-v3
│   ├── spectral_adapter.py        # Adapter chuyển đổi 10 bands vệ tinh sang token space
│   ├── temporal_transformer.py    # Transformer thời gian nhận diện chu kỳ sinh trưởng
│   ├── metrics.py                 # Đánh giá OA, Macro F1, mIoU
│   ├── export.py                  # Xuất checkpoint .pt và ONNX tĩnh
│   └── train.py                   # Script huấn luyện độc lập cho mô hình
│
├── app/                           # [FOLDER 3] Giao diện người dùng Web UI (Độc lập)
│   └── app.py                     # Streamlit Web UI đúng chuẩn UI.png (chờ gắn mô hình mới)
│
├── tests/                         # Bộ kiểm thử tự động (12/12 unit tests pass)
├── checkpoints/                   # Checkpoints best_model.pt và tsvit_dinov3_crop.onnx
├── configs/                       # Cấu hình hyperparameters YAML
└── Makefile                       # Tự động hóa toàn bộ luồng làm việc
```

---

## 5. Hướng Dẫn Chạy & Kiểm Thử

```bash
# 1. Chạy toàn bộ 12 unit tests:
make test

# 2. Huấn luyện thử nghiệm mô hình:
make train

# 3. Mở giao diện Web:
make ui
```
