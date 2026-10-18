"""
Streamlit Web Application for Sentinel-2 Crop Classification & Segmentation:
- Top banner: Satellite Crop Recognition
- Controls card: Latitude, Longitude, Radius, Date, Analyze button
- Left card: Interactive Leaflet satellite map (zoomable across France, click-to-select, soft overlay)
- Right card: Crop classification pie breakdown, confidence indicators, CSV export
"""

import sys
import os
from typing import Optional, List, Dict, Tuple
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium

# Ensure project root & app directory in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_DIR = os.path.dirname(os.path.abspath(__file__))

for p in [PROJECT_ROOT, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

try:
    from app.inference import CropInferenceEngine, CLASS_ID_TO_RGB, CLASS_ID_TO_HEX, CLASS_COLORS
    from app.planetary import fetch_planetary_sentinel2
except Exception:
    from inference import CropInferenceEngine, CLASS_ID_TO_RGB, CLASS_ID_TO_HEX, CLASS_COLORS
    from planetary import fetch_planetary_sentinel2

from src.data.constants import CROP_CLASSES_VI

# Page configuration
st.set_page_config(
    page_title="Nhận diện cây trồng từ ảnh vệ tinh",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Clean CSS (High contrast, bright light theme, zero negative margins)
st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC !important;
        color: #1E293B !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
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

    .input-caption {
        font-size: 12.5px;
        color: #64748B;
        margin-top: 4px;
        margin-bottom: 4px;
    }

    .result-title {
        font-size: 18px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
    }

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


def create_soft_overlay_rgba(pred_map: np.ndarray, alpha_val: int = 175) -> np.ndarray:
    """Creates a smooth, anti-aliased RGBA image from the 24x24 prediction map for soft overlay."""
    H, W = pred_map.shape
    rgb_arr = np.zeros((H, W, 3), dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            cls_id = int(pred_map[r, c])
            rgb_arr[r, c] = CLASS_ID_TO_RGB.get(cls_id, (52, 152, 219))

    # Bilinear upsample to 360x360 for smooth parcel transitions
    img_pil = Image.fromarray(rgb_arr).resize((360, 360), Image.Resampling.BILINEAR)
    r_arr = np.array(img_pil.convert("RGBA"))
    r_arr[:, :, 3] = alpha_val

    # Soft vignette / feathered border 6px
    feather = 6
    for i in range(feather):
        edge_alpha = int(alpha_val * (i + 1) / (feather + 1))
        r_arr[i, :, 3] = np.minimum(r_arr[i, :, 3], edge_alpha)
        r_arr[359 - i, :, 3] = np.minimum(r_arr[359 - i, :, 3], edge_alpha)
        r_arr[:, i, 3] = np.minimum(r_arr[:, i, 3], edge_alpha)
        r_arr[:, 359 - i, 3] = np.minimum(r_arr[:, 359 - i, 3], edge_alpha)

    return r_arr


# Initialize session state variables
if 'lat' not in st.session_state:
    st.session_state.lat = 47.9200
if 'lon' not in st.session_state:
    st.session_state.lon = 2.2500
if 'radius' not in st.session_state:
    st.session_state.radius = 1200
if 'target_date' not in st.session_state:
    st.session_state.target_date = datetime.date(2022, 7, 20)
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analyzed_bbox' not in st.session_state:
    st.session_state.analyzed_bbox = None
if 'analyzed_overlay_rgba' not in st.session_state:
    st.session_state.analyzed_overlay_rgba = None

# Synchronize coordinates from map clicks before rendering input widgets
map_state = st.session_state.get("interactive_france_map")
if map_state and isinstance(map_state, dict):
    clicked = map_state.get("last_clicked") or map_state.get("last_object_clicked")
    if clicked and isinstance(clicked, dict) and "lat" in clicked and "lng" in clicked:
        c_lat = round(float(clicked["lat"]), 4)
        c_lon = round(float(clicked["lng"]), 4)
        if (c_lat, c_lon) != st.session_state.get("last_synced_click"):
            st.session_state.lat = c_lat
            st.session_state.lon = c_lon
            st.session_state.last_synced_click = (c_lat, c_lon)

# Top Header
st.markdown('<div class="top-navy-banner">Nhận diện cây trồng từ ảnh vệ tinh</div>', unsafe_allow_html=True)

# 1. Controls Card
with st.container(border=True):
    c1, c2, c3, c4, c5 = st.columns([1.5, 1.5, 1.3, 1.5, 1.4])
    click_id = st.session_state.get("last_synced_click", (st.session_state.lat, st.session_state.lon))
    with c1:
        lat_in = st.number_input(
            "Vĩ độ",
            value=float(st.session_state.lat),
            format="%.4f",
            min_value=-90.0,
            max_value=90.0,
            step=0.001,
            key=f"in_lat_{click_id}"
        )
    with c2:
        lon_in = st.number_input(
            "Kinh độ",
            value=float(st.session_state.lon),
            format="%.4f",
            min_value=-180.0,
            max_value=180.0,
            step=0.001,
            key=f"in_lon_{click_id}"
        )
    with c3:
        radius_in = st.number_input(
            "Bán kính (m)",
            value=int(st.session_state.radius),
            min_value=300,
            max_value=5000,
            step=100,
            key="in_radius"
        )
    with c4:
        date_in = st.date_input(
            "Thời gian quan sát",
            value=st.session_state.target_date,
            format="DD/MM/YYYY",
            key="in_date"
        )
    with c5:
        st.write("")
        st.write("")
        btn_analyze = st.button("Phân tích", type="primary", use_container_width=True, key="btn_run_analysis")

    # If user typed manually into the inputs
    if abs(lat_in - st.session_state.lat) > 1e-4 or abs(lon_in - st.session_state.lon) > 1e-4:
        st.session_state.lat = lat_in
        st.session_state.lon = lon_in
        st.session_state.last_synced_click = (lat_in, lon_in)
    st.session_state.radius = radius_in
    st.session_state.target_date = date_in

    st.markdown(
        f'<div class="input-caption">Tọa độ: {st.session_state.lat:.4f}, {st.session_state.lon:.4f} · Bán kính: {st.session_state.radius}m · Ngày: {st.session_state.target_date.strftime("%d/%m/%Y")} · Dữ liệu Microsoft Planetary Computer & Bản đồ vệ tinh</div>',
        unsafe_allow_html=True
    )

# Compute bounding box
dlat = st.session_state.radius / 111320.0
dlon = st.session_state.radius / (111320.0 * np.cos(np.radians(st.session_state.lat)) + 1e-6)
curr_bounds = [
    [st.session_state.lat - dlat, st.session_state.lon - dlon],
    [st.session_state.lat + dlat, st.session_state.lon + dlon]
]

# Run analysis ONLY when user explicitly clicks "Phân tích" or on initial cold start
should_analyze = btn_analyze or (st.session_state.analysis_results is None)
if should_analyze:
    with st.spinner("Đang kết nối Microsoft Planetary Computer và phân tích cây trồng..."):
        try:
            p_img, p_meta = fetch_planetary_sentinel2(
                lat=st.session_state.lat,
                lon=st.session_state.lon,
                radius=float(st.session_state.radius),
                target_date_str=st.session_state.target_date.strftime("%Y-%m-%d")
            )
            p_sits, p_doy = CropInferenceEngine.simulate_sits_from_image(p_img)
            res = engine.run_inference(p_sits, p_doy, lat=st.session_state.lat, lon=st.session_state.lon)
            
            st.session_state.analysis_results = res
            st.session_state.analyzed_overlay_rgba = create_soft_overlay_rgba(res['prediction_map'])
            st.session_state.analyzed_bbox = curr_bounds
            if btn_analyze:
                st.success(f"Phân tích hoàn tất: Sentinel-2 ngày {p_meta['acquisition_date']} (phủ mây {p_meta['cloud_cover']}%)")
        except Exception as e:
            st.error(f"Lỗi: {e}")

# 2. Main Content Columns
col_left, col_right = st.columns([1.2, 0.85], gap="small")

# --- LEFT CARD: INTERACTIVE SATELLITE MAP ---
with col_left:
    with st.container(border=True):
        m_head1, m_head2 = st.columns([2, 1.5])
        with m_head1:
            st.markdown('<div style="font-size: 14px; font-weight: 600; color: #1E293B; padding-top: 4px;">Bản đồ vệ tinh tương tác</div>', unsafe_allow_html=True)
        with m_head2:
            show_layer = st.checkbox("Hiển thị lớp nhãn màu", value=True, key="cb_show_overlay")

        m = folium.Map(
            location=[st.session_state.lat, st.session_state.lon],
            zoom_start=14,
            tiles=None,
            control_scale=True
        )

        # Esri World Imagery (Same high-res basemap as crop.loanhviet.dev)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics",
            name="Vệ tinh Esri",
            max_zoom=19,
            control=False
        ).add_to(m)

        # Place names and boundaries overlay
        folium.TileLayer(
            tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
            attr="Esri Labels",
            name="Địa danh & Ranh giới",
            overlay=True,
            control=True
        ).add_to(m)

        # Smooth styled bounding box around chosen area
        folium.Rectangle(
            bounds=curr_bounds,
            color="#22C55E",
            weight=2.5,
            opacity=0.9,
            dash_array="6, 6",
            fill=False,
            tooltip="Khu vực chọn phân tích"
        ).add_to(m)

        # Draggable Center Marker & Pulse Pin
        folium.Marker(
            location=[st.session_state.lat, st.session_state.lon],
            draggable=True,
            icon=folium.Icon(color="green", icon="leaf"),
            tooltip="Kéo thả hoặc nhấp chuột để di chuyển tâm ô",
            popup=f"Tâm: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}"
        ).add_to(m)

        # Soft overlay of crop classification
        if show_layer and st.session_state.analyzed_overlay_rgba is not None and st.session_state.analyzed_bbox is not None:
            ImageOverlay(
                image=st.session_state.analyzed_overlay_rgba,
                bounds=st.session_state.analyzed_bbox,
                opacity=0.70,
                interactive=True,
                zindex=400,
                tooltip="Kết quả phân loại cây trồng"
            ).add_to(m)

        map_data = st_folium(
            m,
            height=460,
            use_container_width=True,
            returned_objects=["last_clicked", "last_object_clicked"],
            key="interactive_france_map"
        )

        # Handle map clicks or marker movements to update parcel position (NO AUTOMATIC INFERENCE, NO RELOAD SPAM)
        clicked_pt = None
        if map_data:
            clicked_pt = map_data.get("last_clicked") or map_data.get("last_object_clicked")
        if clicked_pt and isinstance(clicked_pt, dict) and "lat" in clicked_pt and "lng" in clicked_pt:
            c_lat = round(float(clicked_pt["lat"]), 4)
            c_lon = round(float(clicked_pt["lng"]), 4)
            if abs(c_lat - st.session_state.lat) > 1e-4 or abs(c_lon - st.session_state.lon) > 1e-4:
                st.session_state.lat = c_lat
                st.session_state.lon = c_lon
                st.session_state.last_synced_click = (c_lat, c_lon)
                st.rerun()

        st.markdown(
            '<div class="input-caption" style="margin-top: 6px;">💡 <b>Mẹo:</b> Nhấp chuột bất kỳ đâu trên bản đồ nước Pháp để di chuyển ô vuông và bấm <b>Phân tích</b>. Bạn có thể phóng to thu nhỏ bản đồ tự do.</div>',
            unsafe_allow_html=True
        )


# --- RIGHT CARD: RESULTS & CONFIDENCE ---
with col_right:
    with st.container(border=True):
        st.markdown('<div class="result-title">Kết quả phân tích</div>', unsafe_allow_html=True)

        # Retrieve results
        if st.session_state.analysis_results is not None:
            res = st.session_state.analysis_results
            stats = res['crop_stats']
            non_zero_stats = [s for s in stats if s['percentage'] > 0.0]
            display_stats = non_zero_stats[:4] if non_zero_stats else stats[:4]
            labels = [c['name_vi'].replace('(', '').replace(')', '').strip() for c in display_stats]
            values = [c['percentage'] for c in display_stats]
            confs = [c['confidence'] for c in display_stats]
            colors = [c.get('color', CLASS_ID_TO_HEX.get(c['class_id'], '#3498DB')) for c in display_stats]
        else:
            labels = ['Lúa mì', 'Ngô', 'Cải dầu', 'Đồng cỏ']
            values = [57, 39, 4, 0]
            confs = [81.0, 78.0, 75.0, 0.0]
            colors = ['#F5B041', '#E67E22', '#27AE60', '#2ECC71']

        # Plotly Pie Chart
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
            st.markdown('<div class="conf-header">Cây trồng</div>', unsafe_allow_html=True)
        with h2:
            st.markdown('<div class="conf-header" style="text-align: right;">Độ tự tin</div>', unsafe_allow_html=True)

        # Progress bars
        for name, pct, conf, col in zip(labels, values, confs, colors):
            r1, r2 = st.columns([3, 1])
            with r1:
                st.markdown(f'<span style="background-color: {col}; width: 10px; height: 10px; display: inline-block; border-radius: 2px; margin-right: 6px; vertical-align: middle;"></span> **{name}**', unsafe_allow_html=True)
            with r2:
                conf_text = f"{int(conf)}%" if conf > 0 else "—"
                st.markdown(f'<div style="text-align: right; font-weight: 600; color: #1E293B;">{conf_text}</div>', unsafe_allow_html=True)
            prog_val = min(1.0, max(0.0, conf / 100.0))
            st.progress(prog_val)

        st.write("")
        # Download Results CSV
        csv_rows = ["Cay_Trong,Ty_Le_Phan_Tram,Do_Tu_Tin"]
        for name, pct, conf in zip(labels, values, confs):
            csv_rows.append(f"{name},{pct}%,{conf}%")
        csv_report = "\n".join(csv_rows)

        st.download_button(
            label="Tải kết quả phân tích",
            data=csv_report,
            file_name=f"ket_qua_phan_tich_cay_trong_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_download_csv_file"
        )
