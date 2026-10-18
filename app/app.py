"""
Streamlit Web Application for Sentinel-2 Crop Classification & Segmentation:
- Top banner: Satellite Crop Recognition
- Parameters card: Sentinel-2 Live Streaming, File Upload, Benchmark Presets
- Left card: Full-resolution satellite parcel visualization
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
from PIL import Image, ImageDraw

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

def create_crop_overlay(
    img_pil: Image.Image,
    pred_map: Optional[np.ndarray],
    crop_stats: Optional[list],
    mode: str = "overlay"
) -> Image.Image:
    """Overlays colored semantic segmentation mask and crop name pill tags onto satellite image."""
    if pred_map is None or img_pil is None:
        return img_pil

    W, H = img_pil.size
    rgb_mask = np.zeros((24, 24, 3), dtype=np.uint8)
    for r in range(24):
        for c in range(24):
            cls_id = int(pred_map[r, c])
            rgb_mask[r, c] = CLASS_ID_TO_RGB.get(cls_id, (52, 152, 219))

    mask_pil = Image.fromarray(rgb_mask).resize((W, H), Image.Resampling.NEAREST)

    if mode == "mask":
        return mask_pil
    elif mode == "raw":
        return img_pil

    # Semi-transparent blending (45% alpha)
    blended = Image.blend(img_pil.convert("RGBA"), mask_pil.convert("RGBA"), alpha=0.45).convert("RGBA")
    draw = ImageDraw.Draw(blended)

    # Place text pill tags for dominant crops
    if crop_stats:
        placed_boxes = []
        for crop in crop_stats[:4]:
            cls_id = crop["class_id"]
            pct = crop["percentage"]
            if pct < 3.0 or cls_id == 0:
                continue
            ys, xs = np.where(pred_map == cls_id)
            if len(xs) == 0:
                continue
            cx = int(np.mean(xs) / 24.0 * W)
            cy = int(np.mean(ys) / 24.0 * H)

            name = crop["name_vi"].replace("(", "").replace(")", "").strip()
            label_text = f"{name} {pct:.0f}%"

            tw = len(label_text) * 8 + 18
            th = 22
            x1 = max(10, min(W - tw - 10, cx - tw // 2))
            y1 = max(10, min(H - th - 10, cy - th // 2))

            for bx1, by1, bx2, by2 in placed_boxes:
                if abs(x1 - bx1) < tw and abs(y1 - by1) < th:
                    y1 = min(H - th - 10, by2 + 6)

            placed_boxes.append((x1, y1, x1 + tw, y1 + th))
            col_rgb = CLASS_ID_TO_RGB.get(cls_id, (52, 152, 219))

            # Rounded pill badge with crop color accent
            draw.rounded_rectangle([x1, y1, x1 + tw, y1 + th], radius=4, fill=(23, 37, 61, 220))
            draw.rectangle([x1, y1 + 3, x1 + 4, y1 + th - 3], fill=col_rgb)
            draw.text((x1 + 10, y1 + 4), label_text, fill=(255, 255, 255))

    return blended.convert("RGB")


# Predefined Real Samples
LOCATION_PRESETS = {
    "Cánh đồng ngô Loiret": {
        "lat": 48.2140, "lon": 1.9210, "radius": 1150, "date": datetime.date(2022, 7, 20)
    },
    "Cánh đồng lúa mì Beauce": {
        "lat": 48.1250, "lon": 1.8540, "radius": 1200, "date": datetime.date(2022, 6, 15)
    },
    "Cánh đồng cải dầu Eure": {
        "lat": 48.0920, "lon": 1.7850, "radius": 1100, "date": datetime.date(2022, 5, 10)
    },
    "Vùng nông nghiệp Hà Nội": {
        "lat": 21.0153, "lon": 105.8245, "radius": 1500, "date": datetime.date(2023, 4, 15)
    },
    "Vùng lúa nước An Giang": {
        "lat": 10.3759, "lon": 105.4358, "radius": 1500, "date": datetime.date(2023, 3, 20)
    },
    "Tự nhập tọa độ": {
        "lat": 48.2140, "lon": 1.9210, "radius": 1150, "date": datetime.date(2022, 7, 20)
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
    st.session_state.patch_name = "Cánh đồng ngô Loiret"
if 'last_planetary_params' not in st.session_state:
    st.session_state.last_planetary_params = None

# Top Header
st.markdown('<div class="top-navy-banner">Nhận diện cây trồng từ ảnh vệ tinh</div>', unsafe_allow_html=True)

# 1. Controls Card
with st.container(border=True):
    input_mode = st.selectbox(
        "Phương thức nạp dữ liệu",
        [
            "Ảnh vệ tinh trực tiếp Sentinel-2",
            "Tải ảnh từ máy tính"
        ],
        index=0,
        key="input_mode_selector"
    )

    if input_mode == "Ảnh vệ tinh trực tiếp Sentinel-2":
        loc_choice = st.selectbox(
            "Vị trí cánh đồng quan sát",
            options=list(LOCATION_PRESETS.keys()),
            index=0,
            key="planetary_location_choice"
        )
        preset_data = LOCATION_PRESETS[loc_choice]

        c1, c2, c3, c4, c5 = st.columns([1.5, 1.5, 1.4, 1.8, 1.8])
        with c1:
            lat_in = st.number_input("Vĩ độ", value=float(preset_data["lat"]), format="%.4f", min_value=-90.0, max_value=90.0, step=0.001, key=f"in_lat_{loc_choice}")
        with c2:
            lon_in = st.number_input("Kinh độ", value=float(preset_data["lon"]), format="%.4f", min_value=-180.0, max_value=180.0, step=0.001, key=f"in_lon_{loc_choice}")
        with c3:
            radius_in = st.number_input("Bán kính mét", value=int(preset_data["radius"]), min_value=300, max_value=5000, step=100, key=f"in_rad_{loc_choice}")
        with c4:
            date_in = st.date_input("Thời gian quan sát", value=preset_data["date"], format="DD/MM/YYYY", key=f"in_date_{loc_choice}")
        with c5:
            st.write("")
            st.write("")
            btn_fetch_planetary = st.button("Lấy ảnh và phân tích", type="primary", use_container_width=True, key="btn_planetary_fetch")

        st.markdown(
            f'<div class="input-caption">Tọa độ {lat_in:.4f}, {lon_in:.4f} · Bán kính {radius_in} mét · Ngày {date_in.strftime("%d/%m/%Y")} · Dữ liệu Microsoft Planetary Computer</div>',
            unsafe_allow_html=True
        )

        current_params = f"{loc_choice}_{lat_in:.4f}_{lon_in:.4f}_{radius_in}_{date_in}"
        has_gray_mask = False
        if st.session_state.analysis_results is not None:
            pm = st.session_state.analysis_results.get('prediction_map')
            if pm is not None and (len(np.unique(pm)) == 1 and np.unique(pm)[0] == 0):
                has_gray_mask = True

        should_fetch_planetary = (
            btn_fetch_planetary
            or (st.session_state.get('last_planetary_params') != current_params)
            or (st.session_state.current_display_img is None)
            or has_gray_mask
        )


        if should_fetch_planetary:
            with st.spinner("Đang kết nối Microsoft Planetary Computer và tải ảnh Sentinel-2..."):
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
                    st.session_state.patch_name = f"Sentinel-2 ngày {p_meta['acquisition_date']} phủ mây {p_meta['cloud_cover']}%"
                    st.session_state.analysis_results = engine.run_inference(p_sits, p_doy)
                    st.session_state.active_planetary_loc = loc_choice
                    if btn_fetch_planetary:
                        st.success(f"Đã tải thành công ảnh Sentinel-2 ngày {p_meta['acquisition_date']} với độ phủ mây {p_meta['cloud_cover']}%")
                except Exception as e:
                    st.error(f"Lỗi tải ảnh từ Planetary Computer: {e}")

    elif input_mode == "Tải ảnh từ máy tính":
        up_c1, up_c2, up_c3 = st.columns([3, 1.5, 1.2])
        with up_c1:
            up_file = st.file_uploader(
                "Chọn tệp ảnh vệ tinh hoặc viễn thám",
                type=["tif", "tiff", "png", "jpg", "jpeg", "pt"],
                key="custom_uploader_key"
            )
        with up_c2:
            date_upload = st.date_input("Thời gian quan sát", value=datetime.date(2022, 6, 15), format="DD/MM/YYYY", key="upload_date_field")
        with up_c3:
            st.write("")
            st.write("")
            btn_analyze_upload = st.button("Phân tích ảnh", type="primary", use_container_width=True, key="btn_analyze_up")

        st.markdown(
            '<div class="input-caption">Tự động chuẩn hóa dải màu 16 bit và trích xuất chuỗi thời gian nhận diện cây trồng</div>',
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

# 2. Main Content Columns
col_left, col_right = st.columns([1.2, 0.85], gap="small")

# --- LEFT CARD: SATELLITE PARCEL MAP ---
with col_left:
    with st.container(border=True):
        v_col1, v_col2 = st.columns([1.6, 2.4])
        with v_col1:
            st.markdown('<div style="font-size: 13.5px; font-weight: 600; color: #1E293B; padding-top: 6px;">Hiển thị ảnh vệ tinh</div>', unsafe_allow_html=True)
        with v_col2:
            view_mode = st.selectbox(
                "Chế độ hiển thị",
                ["Dán đè nhãn lên ảnh", "Ảnh vệ tinh gốc", "Bản đồ nhãn màu"],
                index=0,
                label_visibility="collapsed",
                key="display_view_mode"
            )

        mode_key = "overlay"
        if view_mode == "Ảnh vệ tinh gốc":
            mode_key = "raw"
        elif view_mode == "Bản đồ nhãn màu":
            mode_key = "mask"

        if st.session_state.current_display_img is not None:
            pred_map = st.session_state.analysis_results.get('prediction_map') if st.session_state.analysis_results else None
            crop_stats = st.session_state.analysis_results.get('crop_stats') if st.session_state.analysis_results else None
            display_to_show = create_crop_overlay(
                st.session_state.current_display_img,
                pred_map,
                crop_stats,
                mode=mode_key
            )
        else:
            display_to_show = Image.new("RGB", (540, 320), "#234E24")

        st.image(display_to_show, use_container_width=True)

        if st.session_state.analysis_results is not None:
            res = st.session_state.analysis_results
            leg_stats = [s for s in res['crop_stats'] if s['percentage'] > 0.0][:5]
            legend_html = " ".join([
                f'<span style="display:inline-flex;align-items:center;margin-right:12px;font-size:12.5px;color:#334155;">'
                f'<span style="background-color:{s["color"]};width:12px;height:12px;border-radius:2px;margin-right:5px;display:inline-block;"></span>'
                f'<strong>{s["name_vi"]}</strong>&nbsp;{s["percentage"]:.1f}%</span>'
                for s in leg_stats
            ])
            st.markdown(f'<div style="margin-top:6px;margin-bottom:8px;padding:6px 10px;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:6px;display:flex;flex-wrap:wrap;">{legend_html}</div>', unsafe_allow_html=True)

        b_c1, b_c2 = st.columns([1.2, 2.5])
        with b_c1:
            if st.button("Làm mới ảnh", key="btn_reset_selection", use_container_width=True):
                st.session_state.analysis_results = None
                st.session_state.current_sits = None
                st.session_state.current_doy = None
                st.session_state.current_display_img = None
                st.session_state.last_planetary_params = None
                st.rerun()
        with b_c2:
            st.markdown('<div style="padding-top: 6px;" class="selection-badge">Kích thước ảnh: 540 × 320 pixel</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="status-text">Đã hoàn tất phân tích nhận diện cây trồng.</div>',
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
