# TSViT-DINOv3: Phân loại cây trồng chuỗi thời gian vệ tinh Sentinel-2

Dự án tái cấu trúc mô hình **TSViT** từ `DeepSatModels/`, thay thế khối không gian bằng backbone **DINO-v3**, tích hợp pipeline thu thập dữ liệu Sentinel-2 tự động và gán nhãn ground truth **RPG 2022**.

---

## 1. Bảng so sánh chi tiết: Code mới vs Code cũ (`DeepSatModels/`)

| Tiêu chí | Code cũ (`DeepSatModels/`) | Code mới (`crop_classification_tsvit_dinov3/`) | Tệp mã nguồn can thiệp |
| :--- | :--- | :--- | :--- |
| **Spatial Backbone** | `space_transformer` dùng ViT thông thường khởi tạo ngẫu nhiên. | **DINO-v3 Spatial Encoder**: Tái cấu trúc cơ chế self-attention không gian theo kiến trúc DINO-v3 tự giám sát (Self-Supervised Vision Transformer). | [`src/models/dinov3_backbone.py`](file:///home/minh/Documents/Docs/Workspace/Projects/chuyen_de/crop_classification_tsvit_dinov3/src/models/dinov3_backbone.py) |
| **Spectral Adapter** | Linear projection đơn giản cho patch $2 \times 2$. | `SpectralAdapter`: Chiếu 10 dải phổ vệ tinh ($10 \to D$) kết hợp Conv2d + GroupNorm + GELU. | [`src/models/spectral_adapter.py`](file:///home/minh/Documents/Docs/Workspace/Projects/chuyen_de/crop_classification_tsvit_dinov3/src/models/spectral_adapter.py) |
| **Temporal Transformer** | Nằm lẫn trong `TSViTdense.py`, mã hóa DOY phụ thuộc nhánh điều kiện gây lỗi ONNX export. | Tách module độc lập, vector hóa mã hóa Day of Year (DOY) bằng `torch.where`, tương thích 100% ONNX graph tracing. | [`src/models/temporal_transformer.py`](file:///home/minh/Documents/Docs/Workspace/Projects/chuyen_de/crop_classification_tsvit_dinov3/src/models/temporal_transformer.py) |
| **Thu thập dữ liệu** | Không có module crawl; phụ thuộc vào file `.pkl` có sẵn trên ổ cứng tác giả. | Tự động hóa crawl qua **Google Earth Engine** và **Microsoft Planetary Computer STAC API** (miễn phí, không cần thẻ tín dụng). | [`src/data/ingest_gee.py`](file:///home/minh/Documents/Docs/Workspace/Projects/chuyen_de/crop_classification_tsvit_dinov3/src/data/ingest_gee.py)<br>[`src/data/ingest_planetary.py`](file:///home/minh/Documents/Docs/Workspace/Projects/chuyen_de/crop_classification_tsvit_dinov3/src/data/ingest_planetary.py) |
| **Gán nhãn Ground Truth** | Nạp nhãn có sẵn từ file pickle của PASTIS. | Module `RPGProcessor`: Đọc đa giác thửa đất RPG 2022 (Shapefile/GeoPackage/GeoJSON), lọc AOI, ánh xạ mã cây trồng và rasterize khớp lưới ảnh vệ tinh. | [`src/data/rpg_processor.py`](file:///home/minh/Documents/Docs/Workspace/Projects/chuyen_de/crop_classification_tsvit_dinov3/src/data/rpg_processor.py) |
| **Tiền xử lý & Cắt patch** | Tách rời thành script `data2windows.py`. | Tích hợp thành luồng khép kín `SynchronousPatchCrop`: cắt đồng bộ ảnh $[T, 10, 24, 24]$, ngày $[T]$ và nhãn $[24, 24]$. | [`src/data/transforms.py`](file:///home/minh/Documents/Docs/Workspace/Projects/chuyen_de/crop_classification_tsvit_dinov3/src/data/transforms.py)<br>[`src/data/dataset_builder.py`](file:///home/minh/Documents/Docs/Workspace/Projects/chuyen_de/crop_classification_tsvit_dinov3/src/data/dataset_builder.py) |
| **Định dạng dữ liệu** | File `.pkl` cồng kềnh (pickle). | Định dạng chuẩn PyTorch `.pt` và NumPy `.npz` nén, kèm `metadata.csv` và script đẩy Kaggle API. | [`src/data/upload_kaggle.py`](file:///home/minh/Documents/Docs/Workspace/Projects/chuyen_de/crop_classification_tsvit_dinov3/src/data/upload_kaggle.py) |
| **Xuất mô hình** | Chỉ lưu PyTorch `.tar` checkpoint. | Xuất đồng thời checkpoint `.pt` và mô hình **ONNX** hỗ trợ dynamic batch/time axes. | [`src/utils/export.py`](file:///home/minh/Documents/Docs/Workspace/Projects/chuyen_de/crop_classification_tsvit_dinov3/src/utils/export.py) |
| **Giao diện người dùng** | Không có (chỉ dòng lệnh CLI). | Web App tương tác bằng Streamlit chuẩn thiết kế: bản đồ thửa đất, vùng chọn $430 \times 321\text{ px}$, biểu đồ Donut tỷ lệ mùa vụ, thanh độ tự tin. | [`app/app.py`](file:///home/minh/Documents/Docs/Workspace/Projects/chuyen_de/crop_classification_tsvit_dinov3/app/app.py) |
| **Tự động hóa vận hành** | Không có Makefile, script rời rạc. | File `Makefile` duy nhất tự động hóa 100% vòng đời (setup, crawl, test, train, export, UI, docker). | [`Makefile`](file:///home/minh/Documents/Docs/Workspace/Projects/chuyen_de/crop_classification_tsvit_dinov3/Makefile) |

---

## 2. Cấu trúc thư mục tinh gọn (Đã loại bỏ thư mục trùng lặp)

Toàn bộ code trùng lặp (`data_pipeline/` và `model_tsvit_dinov3/`) đã được dọn sạch, tập trung vào kiến trúc `src/`:

```
crop_classification_tsvit_dinov3/
├── Makefile                      # Tự động hóa toàn bộ quy trình 1-click
├── requirements.txt              # Danh sách thư viện tối thiểu
├── Dockerfile                    # Container triển khai
├── train.py                      # CLI huấn luyện & đánh giá mô hình
├── dataset-metadata.json         # Cấu hình đẩy dataset lên Kaggle
├── configs/
│   └── model_config.yaml         # Siêu tham số mô hình TSViT-DINOv3
├── src/
│   ├── data/
│   │   ├── constants.py          # 10 bands S2, μ/σ PASTIS24, bảng nhãn RPG 2022
│   │   ├── transforms.py         # Lọc mây SCL, chuẩn hóa, cắt patch 24x24
│   │   ├── ingest_planetary.py   # Thu thập ảnh qua Microsoft Planetary Computer

│   │   ├── rpg_processor.py      # Đọc polygon RPG 2022 & Rasterize nhãn [H, W]
│   │   ├── dataset_builder.py    # Điều phối tạo dataset hoàn chỉnh (.pt / .npz)
│   │   ├── dataloader.py         # PyTorch Dataset & DataLoader
│   │   └── upload_kaggle.py      # Đóng gói và upload Kaggle qua Kaggle API
│   ├── models/
│   │   ├── spectral_adapter.py   # Conv2D + GroupNorm chiếu 10 kênh sang token
│   │   ├── temporal_transformer.py # Temporal Attention xử lý chuỗi thời gian T
│   │   ├── dinov3_backbone.py    # DINO-v3 Spatial Attention xử lý không gian N
│   │   └── tsvit_dinov3.py       # Mô hình tích hợp hoàn chỉnh [B, 19, 24, 24]
│   └── utils/
│       ├── metrics.py            # Tính OA, Macro F1, mIoU
│       └── export.py             # Xuất checkpoint PyTorch và ONNX runtime
├── app/
│   ├── app.py                    # Streamlit Web UI đúng thiết kế UI.png
│   └── inference.py              # Luồng dự đoán phục vụ Web UI
├── notebooks/
│   └── train_colab.ipynb         # Notebook huấn luyện GPU trên Google Colab
└── tests/
    ├── test_data_pipeline.py     # 6 unit tests kiểm tra pipeline dữ liệu
    └── test_model.py             # 6 unit tests kiểm tra mô hình & export ONNX
```

---

## 3. Hướng dẫn chạy bước lấy dữ liệu (Data Pipeline)

Bạn có thể chọn 1 trong 4 cách lấy dữ liệu tùy theo nhu cầu:

### Bước 1: Kéo dữ liệu vệ tinh thật qua Microsoft Planetary Computer (Miễn phí 100%, không cần tài khoản)
API công khai từ Microsoft Planetary Computer cho phép tải Sentinel-2 L2A qua STAC API đúng năm 2022:
```bash
.venv/bin/python -m src.data.ingest_planetary \
  --bbox 1.50 48.00 2.15 48.45 \
  --start_date 2022-01-01 \
  --end_date 2022-12-01 \
  --cloud_cover_max 20 \
  --output_dir data/raw/planetary_2022
```

---

### Bước 2: Tải Ground Truth RPG-2022 và Cắt Patch Đồng Bộ $24 \times 24$ (40,532 files)

1. **Tải dữ liệu vector RPG-2022 (Shapefile `.shp` hoặc `.gpkg`):**
   - Tải từ cổng dữ liệu mở Pháp: [data.gouv.fr (RPG 2022)](https://www.data.gouv.fr/fr/datasets/registre-parcellaire-graphique-rpg-contours-des-parcelles-et-ilots-culturaux-et-leur-groupe-de-cultures-majoritaire-de-2015-a-2022/) hoặc [IGN Geoservices](https://geoservices.ign.fr/rpg).
   - Đặt file vào thư mục: `data/raw/rpg_2022/`.
2. **Chạy script điều phối cắt patch đồng bộ ảnh $X [T, 10, 24, 24]$ và nhãn $Y [24, 24]$:**
   ```bash
   .venv/bin/python -c "
   from src.data.dataset_builder import S2CropDatasetBuilder
   builder = S2CropDatasetBuilder(patch_size=24)
   # Module tự động đọc ảnh S2, nạp polygon RPG, rasterize và cắt patch
   res = builder.build_synthetic_dataset(num_samples_target=2400, output_dir='data/processed/patches')
   print(res)
   "
   ```

---

## 4. Các lệnh vận hành tự động (Makefile Cheat Sheet)

| Lệnh | Ý nghĩa |
| :--- | :--- |
| `make venv` | Tạo môi trường ảo `.venv` |
| `make install` | Cài đặt các thư viện cần thiết từ `requirements.txt` |
| `make data` | Sinh dữ liệu mẫu thử nghiệm |
| `make test` | Chạy 12 unit tests kiểm thử data pipeline và model |
| `make train` | Huấn luyện mô hình TSViT-DINOv3 (tự xuất checkpoint `.pt` và `.onnx`) |
| `make export` | Xuất mô hình sang định dạng ONNX Runtime |
| `make ui` | Khởi động giao diện Web tương tác Streamlit (`http://localhost:8501`) |
| `make clean` | Dọn dẹp cache, temporary logs và file rác |
