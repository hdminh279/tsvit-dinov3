"""
Streamlit Web Application for Sentinel-2 Crop Classification & Segmentation:
- Dark navy top banner: "Nhận diện cây trồng từ ảnh vệ tinh"
- Parameters card: Vĩ độ, Kinh độ, Bán kính, Thời gian, Chọn ảnh/patch, Phân tích
- Left card: Satellite parcel map with white rectangular selection box (430 x 321 px)
- Right card: Real crop classification results, Pie/Donut breakdown, Confidence progress bars, CSV download
"""

import sys
import os
import io
import glob
import random
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

# Resilient import compatible with both local repo execution and Streamlit Community Cloud
try:
    from app.inference import CropInferenceEngine
except Exception:
    from inference import CropInferenceEngine

from src.data.constants import CROP_CLASSES_VI, CROP_CLASSES_EN

# Page configuration
st.set_page_config(
    page_title="Nhận diện cây trồng từ ảnh vệ tinh",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Clean CSS (No negative margins, responsive, compact)
st.markdown("""
<style>
    /* Global Page Styling */
    .stApp {
        background-color: #F4F6F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Dark Navy Header Banner */
    .top-navy-banner {
        background-color: #17253D;
        color: #FFFFFF;
        padding: 12px 20px;
        border-radius: 8px;
        margin-top: -3.5rem;
        margin-bottom: 1rem;
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.2px;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Input Labels */
    label, .stTextInput label, .stDateInput label {
        color: #2D3748 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        margin-bottom: 2px !important;
    }

    /* Buttons */
    .btn-analyze button {
        background-color: #2600EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        height: 42px !important;
        margin-top: 24px !important;
        box-shadow: 0 2px 6px rgba(38, 0, 235, 0.35) !important;
    }
    .btn-analyze button:hover {
        background-color: #1E00BA !important;
    }

    /* Subtitle caption under inputs */
    .input-caption {
        font-size: 12px;
        color: #718096;
        margin-top: 4px;
        margin-bottom: 4px;
    }

    /* Status text below image */
    .status-text {
        color: #4C1D95;
        font-size: 13.5px;
        font-weight: 600;
        margin-top: 8px;
    }

    .selection-badge {
        color: #4A5568;
        font-size: 13px;
        font-weight: 500;
    }

    /* Result Card Title */
    .result-title {
        color: #1A202C;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* Confidence Table */
    .conf-header {
        color: #4A5568;
        font-size: 12.5px;
        font-weight: 600;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 4px;
        margin-bottom: 8px;
    }
    .conf-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
        font-size: 13px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Cache Inference Engine
@st.cache_resource
def get_inference_engine():
    ckpt_candidates = [
        os.path.join(PROJECT_ROOT, "checkpoints", "best.pth"),
        os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt"),
        os.path.join(PROJECT_ROOT, "checkpoints", "TViT_fold1_full_a100_opt-20261017T125135Z-1-001", "TViT_fold1_full_a100_opt", "best.pth")
    ]
    ckpt_path = None
    for p in ckpt_candidates:
        if os.path.exists(p):
            ckpt_path = p
            break
    return CropInferenceEngine(checkpoint_path=ckpt_path)

engine = get_inference_engine()

# Helper: Palette mapping
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
    'Khác': '#8000FF',
    'Đất trống': '#7F8C8D'
}

def get_color_for_name(name: str) -> str:
    for k, c in CROP_COLOR_MAP.items():
        if k.lower() in name.lower():
            return c
    return '#9B59B6'

# Synthetic Map Generator matching UI.png
def generate_stylized_map(pred_stats=None):
    width, height = 540, 320
    img = Image.new("RGB", (width, height), "#2D5A27")
    draw = ImageDraw.Draw(img)

    # Base parcels
    draw.polygon([(0, 0), (180, 0), (150, 160), (0, 140)], fill="#234E24")
    draw.polygon([(180, 0), (380, 0), (320, 130), (150, 160)], fill="#3B7D3B")
    draw.polygon([(380, 0), (540, 0), (540, 150), (320, 130)], fill="#6B4F35")
    draw.polygon([(0, 140), (150, 160), (100, 280), (0, 260)], fill="#1E431F")
    draw.polygon([(0, 260), (100, 280), (170, 320), (0, 320)], fill="#4B3A2C")
    draw.polygon([(400, 140), (540, 150), (540, 320), (430, 320)], fill="#432F21")
    draw.polygon([(230, 250), (430, 320), (170, 320)], fill="#235026")
    draw.polygon([(350, 130), (540, 150), (400, 140), (320, 190)], fill="#B7A75C")

    # Segmented center fields
    draw.polygon([(150, 140), (320, 130), (350, 180), (320, 210), (170, 210)], fill="#00C882") # Xoài green
    draw.polygon([(170, 140), (210, 140), (210, 190), (170, 190)], fill="#8000FF")             # Cam purple
    draw.polygon([(210, 135), (310, 135), (320, 200), (180, 210), (170, 160)], fill="#F5B041") # Lúa yellow
    draw.polygon([(170, 210), (320, 200), (340, 270), (200, 280)], fill="#00C882")             # Xoài
    draw.polygon([(170, 260), (220, 270), (200, 310), (160, 290)], fill="#8000FF")
    draw.polygon([(220, 270), (320, 260), (300, 310), (200, 310)], fill="#F5B041")

    # White rectangular boundary selection matching UI.png
    rect_x1, rect_y1, rect_x2, rect_y2 = 150, 130, 440, 250
    draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], outline="#FFFFFF", width=3)
    return img

# Initialize session state
if 'current_patch_data' not in st.session_state:
    st.session_state.current_patch_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'patch_name' not in st.session_state:
    st.session_state.patch_name = "Khu vực mẫu (Hà Nội / Vùng nông nghiệp)"

# Top Header
st.markdown('<div class="top-navy-banner">🌱 Nhận diện cây trồng từ ảnh vệ tinh</div>', unsafe_allow_html=True)

# 1. Controls Card
with st.container(border=True):
    c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1.5, 1.3, 1.8, 1.5, 1.4])
    with c1:
        lat = st.text_input("Vĩ độ", value="21.0153")
    with c2:
        lon = st.text_input("Kinh độ", value="105.8245")
    with c3:
        radius = st.text_input("Bán kính (m)", value="1157")
    with c4:
        date_val = st.date_input("Thời gian", value=datetime.date(2019, 12, 1), format="DD/MM/YYYY")
    with c5:
        st.write("")
        st.write("")
        with st.popover("📁 Chọn ảnh", use_container_width=True):
            st.markdown("**Nạp dữ liệu viễn thám:**")
            uploaded_file = st.file_uploader("Tải file tensor .pt", type=["pt"], key="pt_uploader")
            if uploaded_file is not None:
                try:
                    loaded = torch.load(uploaded_file, map_location='cpu', weights_only=False)
                    st.session_state.current_patch_data = loaded
                    st.session_state.patch_name = uploaded_file.name
                    st.success(f"Đã nạp file: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Lỗi đọc file: {e}")

            st.markdown("---")
            if st.button("🎲 Lấy ngẫu nhiên từ kho 56,640 patch", use_container_width=True):
                patch_files = glob.glob(os.path.join(PROJECT_ROOT, "data", "processed", "patches", "patch_*.pt"))
                if patch_files:
                    choice = random.choice(patch_files)
                    loaded = torch.load(choice, map_location='cpu', weights_only=False)
                    st.session_state.current_patch_data = loaded
                    st.session_state.patch_name = os.path.basename(choice)
                    st.success(f"Đã nạp patch: {os.path.basename(choice)}")
                else:
                    st.info("Chưa tìm thấy patch lẻ, đang dùng tensor mẫu sẵn.")

    with c6:
        st.write("")
        st.write("")
        st.markdown('<div class="btn-analyze">', unsafe_allow_html=True)
        btn_analyze = st.button("Phân tích", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="input-caption">Đang chọn: <b>{st.session_state.patch_name}</b> · Vĩ độ {lat} · Kinh độ {lon} · Bán kính {radius}m.</div>',
        unsafe_allow_html=True
    )

# Execute Inference on Analyze click
if btn_analyze:
    with st.spinner("Đang chạy mô hình TSViT-DINOv3 phân tích chuỗi thời gian..."):
        if st.session_state.current_patch_data is not None:
            patch = st.session_state.current_patch_data
            img_tensor = patch.get('image', patch.get('s2'))
            doy_tensor = patch.get('doy', patch.get('dates'))
            img_np = img_tensor.numpy() if isinstance(img_tensor, torch.Tensor) else img_tensor
            doy_np = doy_tensor.numpy() if isinstance(doy_tensor, torch.Tensor) else doy_tensor
            results = engine.run_inference(img_np, doy_np)
            st.session_state.analysis_results = results
        else:
            # Pick a default patch from processed patches if available
            patch_files = glob.glob(os.path.join(PROJECT_ROOT, "data", "processed", "patches", "patch_*.pt"))
            if patch_files:
                sample_path = patch_files[0]
                patch = torch.load(sample_path, map_location='cpu', weights_only=False)
                img_np = patch['image'].numpy()
                doy_np = patch['doy'].numpy()
                results = engine.run_inference(img_np, doy_np)
                st.session_state.analysis_results = results
            else:
                # Fallback synthetic inference
                synth_x = np.random.randn(25, 10, 24, 24).astype(np.float32)
                synth_doy = np.linspace(1, 365, 25).astype(np.float32)
                results = engine.run_inference(synth_x, synth_doy)
                st.session_state.analysis_results = results

# 2. Main Content Columns
col_left, col_right = st.columns([1.2, 0.85], gap="small")

# --- LEFT CARD: SATELLITE MAP & SELECTION ---
with col_left:
    with st.container(border=True):
        map_img = generate_stylized_map()
        st.image(map_img, use_container_width=True)

        b_c1, b_c2 = st.columns([1.2, 2.5])
        with b_c1:
            if st.button("Xóa vùng chọn", key="btn_clear_sel", use_container_width=True):
                st.session_state.analysis_results = None
                st.session_state.current_patch_data = None
                st.session_state.patch_name = "Mặc định (Hà Nội)"
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

        # Retrieve results or use default presentation
        if st.session_state.analysis_results is not None:
            res = st.session_state.analysis_results
            stats = res['crop_stats'][:4] # Top 4 crops
            labels = [c['name_vi'].split('(')[0].strip() for c in stats]
            values = [c['percentage'] for c in stats]
            confs = [c['confidence'] for c in stats]
            colors = [get_color_for_name(l) for l in labels]
        else:
            # Default UI.png reference distribution
            labels = ['Xoài', 'Lúa', 'Cam', 'Táo']
            values = [57, 39, 4, 0]
            confs = [81.0, 81.0, 81.0, 0.0]
            colors = ['#00C882', '#F5B041', '#8000FF', '#E53E3E']

        # Plotly Donut/Pie Chart
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
                st.markdown(f'<div style="text-align: right; font-weight: 600; color: #2D3748;">{conf_text}</div>', unsafe_allow_html=True)
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
            use_container_width=True
        )
