"""
Streamlit Web Application for Sentinel-2 Crop Classification & Segmentation:
- Dark navy top banner: "Nhận diện cây trồng từ ảnh vệ tinh"
- Parameters card: Vĩ độ, Kinh độ, Bán kính, Thời gian, Chọn ảnh mẫu/TIF/PT, Phân tích
- Left card: Satellite parcel map with white rectangular selection box (430 x 321 px)
- Right card: Real crop classification results, Pie/Donut breakdown, Confidence progress bars, CSV download
"""

import sys
import os
import io
import glob
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image, ImageDraw

# Ensure project root & app directory in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_DIR = os.path.dirname(os.path.abspath(__file__))

for p in [PROJECT_ROOT, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

try:
    from app.inference import CropInferenceEngine
    from app.planetary import fetch_planetary_sentinel2
except Exception:
    from inference import CropInferenceEngine
    from planetary import fetch_planetary_sentinel2

from src.data.constants import CROP_CLASSES_VI, CROP_CLASSES_EN

# Page configuration
st.set_page_config(
    page_title="Nhận diện cây trồng từ ảnh vệ tinh",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Clean CSS (High contrast, bright light theme, zero negative margins)
st.markdown("""
<style>
    /* Force Bright Background */
    .stApp {
        background-color: #F8FAFC !important;
        color: #1E293B !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Dark Navy Header Banner */
    .top-navy-banner {
        background-color: #17253D;
        color: #FFFFFF !important;
        padding: 14px 24px;
        border-radius: 8px;
        margin-top: -3.5rem;
        margin-bottom: 1rem;
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.2px;
        display: flex;
        align-items: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }

    /* Subtitle caption under inputs */
    .input-caption {
        font-size: 12.5px;
        color: #64748B;
        margin-top: 4px;
        margin-bottom: 4px;
    }

    /* Status text below image */
    .status-text {
        color: #6B21A8;
        font-size: 13.5px;
        font-weight: 600;
        margin-top: 8px;
    }

    .selection-badge {
        font-size: 13px;
        font-weight: 500;
        color: #475569;
    }

    /* Result Card Title */
    .result-title {
        font-size: 18px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
    }

    /* Confidence Table */
    .conf-header {
        font-size: 13px;
        font-weight: 600;
        color: #475569;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 4px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Cache Inference Engine
@st.cache_resource
def get_inference_engine():
    ckpt_candidates = [
        os.path.join(APP_DIR, "best.pth"),
        os.path.join(PROJECT_ROOT, "checkpoints", "best.pth"),
        os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
    ]
    ckpt_path = None
    for p in ckpt_candidates:
        if os.path.exists(p):
            ckpt_path = p
            break
    return CropInferenceEngine(checkpoint_path=ckpt_path)

engine = get_inference_engine()

# Color mapping for crops
CROP_COLOR_MAP = {
    'Xoài': '#00C882',
    'Lúa': '#F5B041',
    'Lúa mì': '#F5B041',
    'Lúa mì mùa đông': '#F5B041',
    'Ngô (Bắp)': '#E67E22',
    'Cam': '#8000FF',
    'Táo': '#E53E3E',
    'Cải dầu': '#27AE60',
    'Cải dầu vụ đông': '#27AE60',
    'Đồng cỏ': '#2ECC71',
    'Cây ăn quả': '#9B59B6',
    'Khác': '#8000FF',
    'Đất trống': '#7F8C8D'
}

def get_color_for_name(name: str) -> str:
    for k, c in CROP_COLOR_MAP.items():
        if k.lower() in name.lower():
            return c
    return '#3498DB'

# Predefined Real Samples
SAMPLE_PRESETS = {
    "Mẫu 1: Cánh đồng Lúa mì (Winter Wheat)": {
        "pt_path": os.path.join(APP_DIR, "samples", "mau_lua_mi.pt"),
        "lat": "48.1250",
        "lon": "1.8540",
        "radius": "1200",
        "date": datetime.date(2022, 6, 15)
    },
    "Mẫu 2: Cánh đồng Ngô (Corn / Maize)": {
        "pt_path": os.path.join(APP_DIR, "samples", "mau_ngo.pt"),
        "lat": "48.2140",
        "lon": "1.9210",
        "radius": "1150",
        "date": datetime.date(2022, 7, 20)
    },
    "Mẫu 3: Cánh đồng Cải dầu (Winter Rapeseed)": {
        "pt_path": os.path.join(APP_DIR, "samples", "mau_cai_dau.pt"),
        "lat": "48.0920",
        "lon": "1.7850",
        "radius": "1100",
        "date": datetime.date(2022, 5, 10)
    },
    "Mẫu 4: Vườn Cây ăn quả (Orchards)": {
        "pt_path": os.path.join(APP_DIR, "samples", "mau_cay_an_qua.pt"),
        "lat": "48.1560",
        "lon": "1.8920",
        "radius": "1050",
        "date": datetime.date(2022, 8, 5)
    }
}

# Initialize session state variables
if 'current_sits' not in st.session_state:
    st.session_state.current_sits = None
if 'current_doy' not in st.session_state:
    st.session_state.current_doy = None
if 'current_label' not in st.session_state:
    st.session_state.current_label = None
if 'current_display_img' not in st.session_state:
    st.session_state.current_display_img = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'patch_name' not in st.session_state:
    st.session_state.patch_name = "Mẫu 1: Cánh đồng Lúa mì (Winter Wheat)"

# Load default sample on first visit
if st.session_state.current_sits is None:
    def_pt = SAMPLE_PRESETS["Mẫu 1: Cánh đồng Lúa mì (Winter Wheat)"]["pt_path"]
    if os.path.exists(def_pt):
        d = torch.load(def_pt, map_location='cpu', weights_only=False)
        st.session_state.current_sits = d['image'].numpy()
        st.session_state.current_doy = d['doy'].numpy()
        st.session_state.current_label = d['label'].numpy() if 'label' in d else None
        # RGB preview
        mid_t = st.session_state.current_sits.shape[0] // 2
        rgb = st.session_state.current_sits[mid_t, [2, 1, 0], :, :].transpose(1, 2, 0)
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb_norm = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
        st.session_state.current_display_img = Image.fromarray((rgb_norm * 255).astype(np.uint8)).resize((540, 320), Image.Resampling.NEAREST)
        st.session_state.analysis_results = engine.run_inference(
            st.session_state.current_sits,
            st.session_state.current_doy,
            gt_label=st.session_state.current_label
        )

# Top Header
st.markdown('<div class="top-navy-banner">Nhận diện cây trồng từ ảnh vệ tinh</div>', unsafe_allow_html=True)

# 1. Controls Card
with st.container(border=True):
    input_mode = st.radio(
        "Phương thức nạp dữ liệu:",
        [
            "🛰️ Lấy ảnh trực tiếp từ vệ tinh (Microsoft Planetary Computer)",
            "📁 Tải ảnh từ máy tính (TIF, GeoTIFF, PNG, JPG, PT)",
            "🌾 Chọn cánh đồng mẫu (Đã gắn nhãn RPG 2022)"
        ],
        index=0,
        horizontal=True,
        key="input_mode_selector"
    )

    if input_mode == "🛰️ Lấy ảnh trực tiếp từ vệ tinh (Microsoft Planetary Computer)":
        st.markdown("<div style='font-size: 13px; font-weight: 600; color: #475569; margin-bottom: 6px;'>Gợi ý tọa độ cánh đồng tiêu chuẩn (Bấm để điền nhanh):</div>", unsafe_allow_html=True)
        b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
        with b_col1:
            if st.button("🇫🇷 Lúa mì Beauce", key="qc_wheat", use_container_width=True):
                st.session_state.lat_val = 48.1250
                st.session_state.lon_val = 1.8540
                st.session_state.rad_val = 1200
                st.session_state.date_val = datetime.date(2022, 6, 15)
                st.rerun()
        with b_col2:
            if st.button("🇫🇷 Cánh đồng Ngô", key="qc_corn", use_container_width=True):
                st.session_state.lat_val = 48.2140
                st.session_state.lon_val = 1.9210
                st.session_state.rad_val = 1150
                st.session_state.date_val = datetime.date(2022, 7, 20)
                st.rerun()
        with b_col3:
            if st.button("🇫🇷 Cải dầu Eure", key="qc_rapeseed", use_container_width=True):
                st.session_state.lat_val = 48.0920
                st.session_state.lon_val = 1.7850
                st.session_state.rad_val = 1100
                st.session_state.date_val = datetime.date(2022, 5, 10)
                st.rerun()
        with b_col4:
            if st.button("🇻🇳 ĐB Sông Hồng", key="qc_redriver", use_container_width=True):
                st.session_state.lat_val = 21.0153
                st.session_state.lon_val = 105.8245
                st.session_state.rad_val = 1500
                st.session_state.date_val = datetime.date(2023, 4, 15)
                st.rerun()
        with b_col5:
            if st.button("🇻🇳 ĐB Sông Cửu Long", key="qc_mekong", use_container_width=True):
                st.session_state.lat_val = 10.3759
                st.session_state.lon_val = 105.4358
                st.session_state.rad_val = 1500
                st.session_state.date_val = datetime.date(2023, 3, 20)
                st.rerun()

        lat_default = st.session_state.get('lat_val', 48.1250)
        lon_default = st.session_state.get('lon_val', 1.8540)
        rad_default = st.session_state.get('rad_val', 1200)
        date_default = st.session_state.get('date_val', datetime.date(2022, 6, 15))

        c1, c2, c3, c4, c5 = st.columns([1.5, 1.5, 1.4, 1.8, 1.8])
        with c1:
            lat_in = st.number_input("Vĩ độ (Lat)", value=float(lat_default), format="%.4f", min_value=-90.0, max_value=90.0, step=0.001, key="in_lat")
        with c2:
            lon_in = st.number_input("Kinh độ (Lon)", value=float(lon_default), format="%.4f", min_value=-180.0, max_value=180.0, step=0.001, key="in_lon")
        with c3:
            radius_in = st.number_input("Bán kính (m)", value=int(rad_default), min_value=300, max_value=5000, step=100, help="Khuyên dùng 1000m - 1500m (khoảng 2.4km vuông) để thấy trọn vẹn cánh đồng", key="in_rad")
        with c4:
            date_in = st.date_input("Thời gian quan sát", value=date_default, format="DD/MM/YYYY", key="in_date")
        with c5:
            st.write("")
            st.write("")
            btn_fetch_planetary = st.button("Lấy ảnh vệ tinh & Phân tích", type="primary", use_container_width=True, key="btn_planetary_fetch")

        st.markdown(
            f'<div class="input-caption">Đang chọn: Vĩ độ <b>{lat_in:.4f}</b> · Kinh độ <b>{lon_in:.4f}</b> · Bán kính <b>{radius_in}m</b> · Ngày <b>{date_in.strftime("%d/%m/%Y")}</b>. Hệ thống tự động tìm ảnh Sentinel-2 không mây từ Microsoft Planetary Computer.</div>',
            unsafe_allow_html=True
        )

        if btn_fetch_planetary:
            with st.spinner(f"Đang kết nối Microsoft Planetary Computer... Tải ảnh Sentinel-2 L2A tại ({lat_in:.4f}, {lon_in:.4f})..."):
                try:
                    p_img, p_meta = fetch_planetary_sentinel2(
                        lat=lat_in,
                        lon=lon_in,
                        radius=float(radius_in),
                        target_date_str=date_in.strftime("%Y-%m-%d")
                    )
                    p_sits, p_doy = CropInferenceEngine.simulate_sits_from_image(p_img)
                    st.session_state.current_display_img = p_img
                    st.session_state.current_sits = p_sits
                    st.session_state.current_doy = p_doy
                    st.session_state.current_label = None
                    st.session_state.patch_name = f"Sentinel-2 Live ({p_meta['acquisition_date']}, mây {p_meta['cloud_cover']}%)"
                    st.session_state.analysis_results = engine.run_inference(p_sits, p_doy)
                    st.success(f"Đã tải thành công ảnh vệ tinh Sentinel-2 ngày {p_meta['acquisition_date']} (Độ phủ mây: {p_meta['cloud_cover']}%)!")
                except Exception as e:
                    st.error(f"Lỗi tải ảnh từ Planetary Computer: {e}")

    elif input_mode == "📁 Tải ảnh từ máy tính (TIF, GeoTIFF, PNG, JPG, PT)":
        # Mode 2: Only file uploader + date (No lat/lon/radius!)
        up_c1, up_c2, up_c3 = st.columns([3, 1.5, 1.2])
        with up_c1:
            up_file = st.file_uploader(
                "Chọn file ảnh vệ tinh hoặc TIF viễn thám (.tif, .png, .jpg, .pt):",
                type=["tif", "tiff", "png", "jpg", "jpeg", "pt"],
                key="custom_uploader_key"
            )
        with up_c2:
            date_upload = st.date_input("Thời gian thu nhận (Mùa vụ)", value=datetime.date(2022, 6, 15), format="DD/MM/YYYY", key="upload_date_field")
        with up_c3:
            st.write("")
            st.write("")
            btn_analyze_upload = st.button("Phân tích ảnh", type="primary", use_container_width=True, key="btn_analyze_up")

        st.markdown(
            '<div class="input-caption">Chế độ tải ảnh từ máy tính: Không cần nhập vĩ độ/kinh độ/bán kính. Ảnh sẽ tự động được giãn tương phản 16-bit sắc nét và đưa vào mô hình.</div>',
            unsafe_allow_html=True
        )

        if (up_file is not None and st.session_state.patch_name != up_file.name) or btn_analyze_upload:
            if up_file is not None:
                fname = up_file.name.lower()
                if fname.endswith(".pt"):
                    loaded = torch.load(up_file, map_location='cpu', weights_only=False)
                    st.session_state.current_sits = loaded.get('image', loaded.get('s2')).numpy()
                    st.session_state.current_doy = loaded.get('doy', loaded.get('dates')).numpy()
                    st.session_state.current_label = loaded.get('label', None)
                    if st.session_state.current_label is not None and hasattr(st.session_state.current_label, 'numpy'):
                        st.session_state.current_label = st.session_state.current_label.numpy()
                    mid_t = st.session_state.current_sits.shape[0] // 2
                    rgb = st.session_state.current_sits[mid_t, [2, 1, 0], :, :].transpose(1, 2, 0)
                    p2, p98 = np.percentile(rgb, (2, 98))
                    rgb_norm = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
                    st.session_state.current_display_img = Image.fromarray((rgb_norm * 255).astype(np.uint8)).resize((540, 320), Image.Resampling.NEAREST)
                    st.session_state.patch_name = up_file.name
                    st.session_state.analysis_results = engine.run_inference(
                        st.session_state.current_sits,
                        st.session_state.current_doy,
                        gt_label=st.session_state.current_label
                    )
                else:
                    bytes_data = up_file.read()
                    disp_pil, sits, doy = CropInferenceEngine.read_any_image_file(bytes_data)
                    st.session_state.current_sits = sits
                    st.session_state.current_doy = doy
                    st.session_state.current_label = None
                    st.session_state.current_display_img = disp_pil.resize((540, 320), Image.Resampling.NEAREST)
                    st.session_state.patch_name = up_file.name
                    st.session_state.analysis_results = engine.run_inference(sits, doy)

    else:  # Preset mode
        p_c1, p_c2 = st.columns([3, 1.2])
        with p_c1:
            preset_choice = st.selectbox(
                "Chọn cánh đồng vệ tinh mẫu:",
                options=list(SAMPLE_PRESETS.keys()),
                index=0,
                key="source_preset_select"
            )
        with p_c2:
            st.write("")
            st.write("")
            btn_run_preset = st.button("Xem & Phân tích", type="primary", use_container_width=True, key="btn_run_preset")

        preset_info = SAMPLE_PRESETS[preset_choice]
        st.markdown(
            f'<div class="input-caption">Đang chọn: <b>{preset_choice}</b> · Vĩ độ {preset_info["lat"]} · Kinh độ {preset_info["lon"]} · Bán kính {preset_info["radius"]}m · Ngày {preset_info["date"].strftime("%d/%m/%Y")} · Gắn nhãn chuẩn RPG 2022 Cadastre.</div>',
            unsafe_allow_html=True
        )

        if st.session_state.patch_name != preset_choice or btn_run_preset or st.session_state.analysis_results is None:
            st.session_state.patch_name = preset_choice
            pt_file = preset_info["pt_path"]
            if os.path.exists(pt_file):
                d = torch.load(pt_file, map_location='cpu', weights_only=False)
                st.session_state.current_sits = d['image'].numpy()
                st.session_state.current_doy = d['doy'].numpy()
                st.session_state.current_label = d['label'].numpy() if 'label' in d else None
                mid_t = st.session_state.current_sits.shape[0] // 2
                rgb = st.session_state.current_sits[mid_t, [2, 1, 0], :, :].transpose(1, 2, 0)
                p2, p98 = np.percentile(rgb, (2, 98))
                rgb_norm = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
                st.session_state.current_display_img = Image.fromarray((rgb_norm * 255).astype(np.uint8)).resize((540, 320), Image.Resampling.NEAREST)
                st.session_state.analysis_results = engine.run_inference(
                    st.session_state.current_sits,
                    st.session_state.current_doy,
                    gt_label=st.session_state.current_label
                )



# 2. Main Content Columns
col_left, col_right = st.columns([1.2, 0.85], gap="small")

# --- LEFT CARD: SATELLITE MAP & SELECTION ---
with col_left:
    with st.container(border=True):
        if st.session_state.current_display_img is not None:
            display_to_show = st.session_state.current_display_img.copy()
            draw = ImageDraw.Draw(display_to_show)
            draw.rectangle([140, 50, 410, 270], outline="#FFFFFF", width=3)
        else:
            # Default placeholder image
            display_to_show = Image.new("RGB", (540, 320), "#234E24")
            draw = ImageDraw.Draw(display_to_show)
            draw.rectangle([140, 50, 410, 270], outline="#FFFFFF", width=3)

        st.image(display_to_show, use_container_width=True)

        b_c1, b_c2 = st.columns([1.2, 2.5])
        with b_c1:
            if st.button("Xóa vùng chọn", key="btn_reset_selection", use_container_width=True):
                st.session_state.analysis_results = None
                st.session_state.current_sits = None
                st.session_state.current_doy = None
                st.session_state.current_display_img = None
                st.session_state.patch_name = "Khu vực mẫu"
                st.rerun()
        with b_c2:
            st.markdown('<div style="padding-top: 6px;" class="selection-badge">Vùng chọn: 430 × 321 px</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="status-text">Đã phân tích xong vùng chọn (430 × 321 px).</div>',
            unsafe_allow_html=True
        )

# --- RIGHT CARD: RESULTS & CONFIDENCE ---
with col_right:
    with st.container(border=True):
        st.markdown('<div class="result-title">Kết quả</div>', unsafe_allow_html=True)

        # Retrieve results
        if st.session_state.analysis_results is not None:
            res = st.session_state.analysis_results
            stats = res['crop_stats'][:4]
            labels = [c['name_vi'].split('(')[0].strip() for c in stats]
            values = [c['percentage'] for c in stats]
            confs = [c['confidence'] for c in stats]
            colors = [get_color_for_name(l) for l in labels]
        else:
            # Default presentation
            labels = ['Lúa mì', 'Ngô', 'Cải dầu', 'Đồng cỏ']
            values = [57, 39, 4, 0]
            confs = [81.0, 78.0, 75.0, 0.0]
            colors = ['#F5B041', '#E67E22', '#27AE60', '#2ECC71']

        # Plotly Pie Chart (Bright, Transparent Background)
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=[v if v > 0 else 0.01 for v in values],
            hole=0.0,
            marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
            textinfo='label+percent',
            textposition='inside',
            insidetextorientation='horizontal',
            hoverinfo='label+percent',
            direction='clockwise',
            sort=False
        )])
        fig.update_layout(
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            height=210,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="sans-serif", size=13, color="#FFFFFF")
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Confidence Table Header
        h1, h2 = st.columns([3, 1])
        with h1:
            st.markdown('<div class="conf-header">Nhãn</div>', unsafe_allow_html=True)
        with h2:
            st.markdown('<div class="conf-header" style="text-align: right;">Độ tự tin</div>', unsafe_allow_html=True)

        # Progress bars
        for name, pct, conf, col in zip(labels, values, confs, colors):
            r1, r2 = st.columns([3, 1])
            with r1:
                st.markdown(f'<span style="color: {col}; font-size: 15px;">■</span> **{name}**', unsafe_allow_html=True)
            with r2:
                conf_text = f"{int(conf)}%" if conf > 0 else "—"
                st.markdown(f'<div style="text-align: right; font-weight: 600; color: #1E293B;">{conf_text}</div>', unsafe_allow_html=True)
            prog_val = min(1.0, max(0.0, conf / 100.0))
            st.progress(prog_val)

        st.write("")
        # Download Results CSV
        csv_rows = ["Nhan,Ty_Le_Phan_Tram,Do_Tu_Tin"]
        for name, pct, conf in zip(labels, values, confs):
            csv_rows.append(f"{name},{pct}%,{conf}%")
        csv_report = "\n".join(csv_rows)

        st.download_button(
            label="Tải kết quả",
            data=csv_report,
            file_name=f"ket_qua_phan_tich_cay_trong_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_download_csv_file"
        )
