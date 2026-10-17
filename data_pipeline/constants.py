"""
Constants for Sentinel-2 satellite time-series data processing and crop taxonomy.
Inherited from DeepSatModels (PASTIS24 and France benchmarks).
"""

import numpy as np
import torch

# 10 Sentinel-2 bands indexing (10m & 20m resampled to 10m)
# Order: B02(Blue), B03(Green), B04(Red), B05(VRE1), B06(VRE2), B07(VRE3), B08(NIR), B8A(NarrowNIR), B11(SWIR1), B12(SWIR2)
S2_10BANDS = [
    'B02', 'B03', 'B04', 'B05', 'B06',
    'B07', 'B08', 'B8A', 'B11', 'B12'
]

BAND_2_IDX = {band: i for i, band in enumerate(S2_10BANDS)}

# Normalization parameters inherited from DeepSatModels PASTIS24 Fold-1
# Applied as (x - mean) / std per channel
S2_MEAN_10B = np.array([
    1165.9398193359375,
    1375.6534423828125,
    1429.2191162109375,
    1764.798828125,
    2719.273193359375,
    3063.61181640625,
    3205.90185546875,
    3319.109619140625,
    2422.904296875,
    1639.370361328125
], dtype=np.float32)

S2_STD_10B = np.array([
    1942.6156005859375,
    1881.9234619140625,
    1959.3798828125,
    1867.2239990234375,
    1754.5850830078125,
    1769.4046630859375,
    1784.860595703125,
    1767.7100830078125,
    1458.963623046875,
    1299.2833251953125
], dtype=np.float32)

DOY_DIVISOR = 365.0001
CROP_PATCH_SIZE = 24  # Square patch 24x24 px

# Crop categories (19 classes taxonomy matching PASTIS24 benchmark & RPG 2022)
CROP_CLASSES_EN = {
    0: "Background / Non-agricultural",
    1: "Meadow / Permanent grassland",
    2: "Soft winter wheat",
    3: "Corn / Maize",
    4: "Winter barley",
    5: "Winter rapeseed",
    6: "Spring barley",
    7: "Sunflower",
    8: "Soybean",
    9: "Peas",
    10: "Protein crops",
    11: "Soy",
    12: "Vegetables / Flowers",
    13: "Vines / Vineyard",
    14: "Fruit trees / Orchards",
    15: "Fallow",
    16: "Tuber crops (Potato)",
    17: "Sugar beet",
    18: "Other cereals"
}

CROP_CLASSES_VI = {
    0: "Đất trống / Khác",
    1: "Đồng cỏ / Cây thức ăn gia súc",
    2: "Lúa mì mùa đông",
    3: "Ngô (Bắp)",
    4: "Đại mạch vụ đông",
    5: "Cải dầu vụ đông",
    6: "Đại mạch vụ xuân",
    7: "Hướng dương",
    8: "Đậu tương (Đậu nành)",
    9: "Đậu Hà Lan",
    10: "Cây họ đậu lấy hạt",
    11: "Đậu khác",
    12: "Rau màu / Hoa",
    13: "Vườn nho",
    14: "Cây ăn quả (Táo, Cam, Xoài)",
    15: "Đất hoang hóa",
    16: "Cây lấy củ (Khoai tây)",
    17: "Củ cải đường",
    18: "Ngũ cốc khác (Lúa nước, Yến mạch)"
}

NUM_CLASSES = len(CROP_CLASSES_EN)

# RPG 2022 crop group code mapping (Registre Parcellaire Graphique)
RPG_2022_GROUP_MAP = {
    '1': 2,    # Blé tendre -> Soft winter wheat
    '2': 3,    # Maïs -> Corn
    '3': 4,    # Orge -> Barley
    '4': 18,   # Autres céréales
    '5': 5,    # Colza -> Rapeseed
    '6': 7,    # Tournesol -> Sunflower
    '7': 11,   # Autres oléagineux
    '8': 9,    # Protéagineux -> Peas / Protein
    '9': 8,    # Soja -> Soybean
    '11': 14,  # Fruits / Vergers -> Orchards
    '12': 13,  # Vignes -> Vineyard
    '14': 12,  # Légumes -> Vegetables
    '15': 16,  # Pommes de terre -> Potato
    '16': 17,  # Betterave -> Sugar beet
    '18': 1,   # Prairies permanentes -> Meadow
    '19': 1,   # Prairies temporaires
    '24': 15,  # Jachères -> Fallow
}
