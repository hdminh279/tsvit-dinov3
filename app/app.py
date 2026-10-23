"""
Streamlit Web Application for Sentinel-2 Crop Classification & Segmentation
Production UI: Clean, high-contrast, bold-bordered dashboard.
- Strict SINGLE 2px border on all inputs (NO nested or double borders).
- Interactive map tuned to 470px height (fully visible on screen with zero cutoff).
- Results card with balanced 180px solid pie chart and neat crop table spacing.
"""

import sys
import os
import math
import datetime
import numpy as np
import pandas as pd
import streamlit as st
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

import torch  # noqa: F401

try:
    from app.inference import CropInferenceEngine, CLASS_ID_TO_RGB, CLASS_ID_TO_HEX, CLASS_COLORS  # noqa: F401
    from app.planetary import fetch_planetary_sentinel2
except Exception:
    from inference import CropInferenceEngine, CLASS_ID_TO_RGB, CLASS_ID_TO_HEX, CLASS_COLORS  # noqa: F401
    from planetary import fetch_planetary_sentinel2

from src.data.constants import CROP_CLASSES_VI  # noqa: F401

st.set_page_config(
    page_title="Nhận diện cây trồng từ ảnh vệ tinh",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Load external CSS stylesheet
# ---------------------------------------------------------------------------
css_file = os.path.join(APP_DIR, "style.css")
if os.path.exists(css_file):
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def get_inference_engine():
    ckpt_candidates = [
        os.path.join(APP_DIR, "best.pth"),
        os.path.join(PROJECT_ROOT, "checkpoints", "best.pth"),
        os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt"),
    ]
    ckpt_path = None
    for p in ckpt_candidates:
        if os.path.exists(p):
            ckpt_path = p
            break
    return CropInferenceEngine(checkpoint_path=ckpt_path)


engine = get_inference_engine()


def create_soft_overlay_rgba(pred_map: np.ndarray, alpha_val: int = 175) -> np.ndarray:
    """Creates a smooth, anti-aliased RGBA image from the prediction map for soft overlay."""
    H, W = pred_map.shape
    rgb_arr = np.zeros((H, W, 3), dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            cls_id = int(pred_map[r, c])
            rgb_arr[r, c] = CLASS_ID_TO_RGB.get(cls_id, (52, 152, 219))

    img_pil = Image.fromarray(rgb_arr).resize((360, 360), Image.Resampling.BILINEAR)
    r_arr = np.array(img_pil.convert("RGBA"))
    r_arr[:, :, 3] = alpha_val

    feather = 6
    for i in range(feather):
        edge_alpha = int(alpha_val * (i + 1) / (feather + 1))
        r_arr[i, :, 3] = np.minimum(r_arr[i, :, 3], edge_alpha)
        r_arr[359 - i, :, 3] = np.minimum(r_arr[359 - i, :, 3], edge_alpha)
        r_arr[:, i, 3] = np.minimum(r_arr[:, i, 3], edge_alpha)
        r_arr[:, 359 - i, 3] = np.minimum(r_arr[:, 359 - i, 3], edge_alpha)

    return r_arr


def build_result_html(labels, values, confs, colors) -> str:
    """Builds the solid circular pie chart + crop breakdown list as clean HTML."""
    total = sum(v for v in values if v > 0) or 1.0
    stops, pie_labels, acc = [], "", 0.0
    for name, v, col in zip(labels, values, colors):
        if v <= 0:
            continue
        start = acc / total
        acc += v
        end = acc / total
        stops.append(f"{col} {start * 100:.2f}% {end * 100:.2f}%")
        share = (end - start) * 100
        if share >= 6:
            ang = (start + end) / 2 * 2 * math.pi
            r = 0.32
            x = 50 + r * 100 * math.sin(ang)
            y = 50 - r * 100 * math.cos(ang)
            name_html = f"{name}<br>" if share >= 12 else ""
            pie_labels += f'<div class="pie-label" style="left:{x:.1f}%;top:{y:.1f}%">{name_html}<b>{share:.1f}%</b></div>'
    gradient = ", ".join(stops) if stops else "#E2E8F0 0% 100%"

    rows = ""
    for name, conf, col in zip(labels, confs, colors):
        conf_txt = f"{int(conf)}%" if conf > 0 else "—"
        rows += (
            f'<div class="crop-row"><div class="crop-top">'
            f'<span class="crop-name"><span class="dot" style="background:{col}"></span>{name}</span>'
            f'<span class="crop-conf">{conf_txt}</span></div>'
            f'<div class="bar"><div style="width:{min(100, max(0, conf)):.0f}%;background:{col}"></div></div></div>'
        )

    return (
        '<div class="result-wrap">'
        '<div class="result-top-group">'
        '<div class="result-title">Kết quả phân tích</div>'
        '<div class="pie-container">'
        f'<div class="pie-chart" style="background: conic-gradient({gradient});">{pie_labels}</div>'
        '</div>'
        '<div class="crop-head"><span>Cây trồng</span><span>Độ tự tin</span></div>'
        f'<div class="crop-list">{rows}</div>'
        '</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "lat": 47.9200,
    "lon": 2.2500,
    "radius": 1200,
    "target_date": datetime.date(2022, 7, 20),
    "analysis_results": None,
    "analyzed_bbox": None,
    "analyzed_overlay_rgba": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sync coordinates from map clicks before rendering input widgets
map_state = st.session_state.get("interactive_france_map")
if map_state and isinstance(map_state, dict):
    clicked = map_state.get("last_clicked")
    if clicked and isinstance(clicked, dict) and "lat" in clicked and "lng" in clicked:
        c_lat = round(float(clicked["lat"]), 4)
        c_lon = round(float(clicked["lng"]), 4)
        if (c_lat, c_lon) != st.session_state.get("last_synced_click"):
            st.session_state.lat = c_lat
            st.session_state.lon = c_lon
            st.session_state.last_synced_click = (c_lat, c_lon)

# ---------------------------------------------------------------------------
# Top Banner (Distinctly separated from controls)
# ---------------------------------------------------------------------------
st.markdown('<div class="top-navy-banner">Nhận diện cây trồng từ ảnh vệ tinh</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Controls Card
# ---------------------------------------------------------------------------
with st.container(key="controls_card"):
    c1, c2, c3, c4, c5 = st.columns([1.3, 1.3, 1.3, 1.6, 1.2])
    click_id = st.session_state.get("last_synced_click", (st.session_state.lat, st.session_state.lon))
    with c1:
        lat_in = st.number_input(
            "Vĩ độ", value=float(st.session_state.lat), format="%.4f",
            min_value=-90.0, max_value=90.0, step=0.001, key=f"in_lat_{click_id}",
        )
    with c2:
        lon_in = st.number_input(
            "Kinh độ", value=float(st.session_state.lon), format="%.4f",
            min_value=-180.0, max_value=180.0, step=0.001, key=f"in_lon_{click_id}",
        )
    with c3:
        radius_in = st.number_input(
            "Bán kính (m)", value=int(st.session_state.radius),
            min_value=300, max_value=5000, step=100, key="in_radius",
        )
    with c4:
        date_in = st.date_input(
            "Thời gian quan sát", value=st.session_state.target_date,
            format="DD/MM/YYYY", key="in_date",
        )
    with c5:
        btn_analyze = st.button("Phân tích", type="primary", use_container_width=True, key="btn_run_analysis")

    if abs(lat_in - st.session_state.lat) > 1e-4 or abs(lon_in - st.session_state.lon) > 1e-4:
        st.session_state.lat = lat_in
        st.session_state.lon = lon_in
        st.session_state.last_synced_click = (lat_in, lon_in)
    st.session_state.radius = radius_in
    st.session_state.target_date = date_in

# Bounding box
dlat = st.session_state.radius / 111320.0
dlon = st.session_state.radius / (111320.0 * np.cos(np.radians(st.session_state.lat)) + 1e-6)
curr_bounds = [
    [st.session_state.lat - dlat, st.session_state.lon - dlon],
    [st.session_state.lat + dlat, st.session_state.lon + dlon],
]

# Run analysis on button click or cold start
should_analyze = btn_analyze or (st.session_state.analysis_results is None)
if should_analyze:
    with st.spinner("Đang kết nối Microsoft Planetary Computer và phân tích cây trồng..."):
        try:
            p_img, p_meta = fetch_planetary_sentinel2(
                lat=st.session_state.lat,
                lon=st.session_state.lon,
                radius=float(st.session_state.radius),
                target_date_str=st.session_state.target_date.strftime("%Y-%m-%d"),
            )
            p_sits, p_doy = CropInferenceEngine.simulate_sits_from_image(p_img)
            res = engine.run_inference(p_sits, p_doy, lat=st.session_state.lat, lon=st.session_state.lon)

            st.session_state.analysis_results = res
            st.session_state.analyzed_overlay_rgba = create_soft_overlay_rgba(res["prediction_map"])
            st.session_state.analyzed_bbox = curr_bounds
            if btn_analyze:
                st.toast(f"Hoàn tất: Sentinel-2 ngày {p_meta['acquisition_date']} (phủ mây {p_meta['cloud_cover']}%)", icon="✅")
        except Exception as e:
            st.toast(f"Lỗi: {e}", icon="⚠️")

# ---------------------------------------------------------------------------
# Main Row: Map WIDER & TALLER (2.35), Results NARROWER (1.0)
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([2.35, 1.0], gap="small")

# --- LEFT: SATELLITE MAP (Tuned 470px height) ------------------------------
with col_left:
    with st.container(key="map_card"):
        st.markdown('<div class="map-title">Bản đồ vệ tinh tương tác</div>', unsafe_allow_html=True)
        show_layer = st.checkbox("Hiển thị lớp nhãn màu", value=True, key="cb_show_overlay")

        m = folium.Map(
            location=[st.session_state.lat, st.session_state.lon],
            zoom_start=14,
            tiles=None,
            control_scale=True,
        )

        m.get_root().header.add_child(folium.Element("""
        <style>
        html, body, #root, #parent, .float-container, .float-child, #map_div, .leaflet-container {
            width: 100% !important;
            height: 100% !important;
            min-height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
        }
        </style>
        """))

        # Guarantee zero white gap by enforcing map container height (470px) and calling invalidateSize
        m.get_root().html.add_child(folium.Element("""
        <script>
        function fitMap() {
            window.dispatchEvent(new Event('resize'));
            var mapEl = document.getElementById('map_div');
            if (mapEl) {
                mapEl.style.height = '470px';
                mapEl.style.width = '100%';
            }
            if (window.map && typeof window.map.invalidateSize === 'function') {
                window.map.invalidateSize();
            }
            if (window.map_div && typeof window.map_div.invalidateSize === 'function') {
                window.map_div.invalidateSize();
            }
        }
        window.addEventListener('load', function() {
            fitMap();
            setTimeout(fitMap, 50);
            setTimeout(fitMap, 200);
            setTimeout(fitMap, 500);
        });
        </script>
        """))

        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics",
            name="Vệ tinh Esri",
            max_zoom=19,
            control=False,
        ).add_to(m)

        folium.TileLayer(
            tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
            attr="Esri Labels",
            name="Địa danh & Ranh giới",
            overlay=True,
            control=True,
        ).add_to(m)

        folium.Rectangle(
            bounds=curr_bounds,
            color="#22C55E",
            weight=2.5,
            opacity=0.9,
            dash_array="6, 6",
            fill=False,
            tooltip="Khu vực chọn phân tích",
        ).add_to(m)

        folium.Marker(
            location=[st.session_state.lat, st.session_state.lon],
            icon=folium.Icon(color="green", icon="leaf"),
            tooltip="Tâm vị trí đã chọn",
            popup=f"Tâm: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}",
        ).add_to(m)

        if show_layer and st.session_state.analyzed_overlay_rgba is not None and st.session_state.analyzed_bbox is not None:
            ImageOverlay(
                image=st.session_state.analyzed_overlay_rgba,
                bounds=st.session_state.analyzed_bbox,
                opacity=0.70,
                interactive=True,
                zindex=400,
                tooltip="Kết quả phân loại cây trồng",
            ).add_to(m)

        map_data = st_folium(
            m,
            height=470,
            use_container_width=True,
            returned_objects=["last_clicked"],
            key="interactive_france_map",
        )

        clicked_pt = map_data.get("last_clicked") if map_data else None
        if clicked_pt and isinstance(clicked_pt, dict) and "lat" in clicked_pt and "lng" in clicked_pt:
            c_lat = round(float(clicked_pt["lat"]), 4)
            c_lon = round(float(clicked_pt["lng"]), 4)
            if abs(c_lat - st.session_state.lat) > 1e-4 or abs(c_lon - st.session_state.lon) > 1e-4:
                st.session_state.lat = c_lat
                st.session_state.lon = c_lon
                st.session_state.last_synced_click = (c_lat, c_lon)
                st.rerun()

        st.markdown(
            '<div class="map-tip">💡 <b>Mẹo:</b> Nhấp chuột bất kỳ đâu trên bản đồ để di chuyển ô vuông, '
            'sau đó bấm <b>Phân tích</b>. Có thể phóng to/thu nhỏ tự do.</div>',
            unsafe_allow_html=True,
        )

# --- RIGHT: RESULTS (Tuned 470px height) -----------------------------------
with col_right:
    with st.container(key="result_card"):
        if st.session_state.analysis_results is not None:
            res = st.session_state.analysis_results
            stats = res["crop_stats"]
            non_zero_stats = [s for s in stats if s["percentage"] > 0.0]
            display_stats = (non_zero_stats or stats)[:5]
            labels = [c["name_vi"].replace("(", "").replace(")", "").strip() for c in display_stats]
            values = [float(c["percentage"]) for c in display_stats]
            confs = [float(c["confidence"]) for c in display_stats]
            colors = [c.get("color", CLASS_ID_TO_HEX.get(c["class_id"], "#3498DB")) for c in display_stats]
        else:
            labels = ["Lúa mì", "Ngô", "Cải dầu", "Đồng cỏ"]
            values = [57.0, 39.0, 4.0, 0.0]
            confs = [81.0, 78.0, 75.0, 0.0]
            colors = ["#F5B041", "#E67E22", "#27AE60", "#2ECC71"]

        st.markdown(
            build_result_html(labels, values, confs, colors),
            unsafe_allow_html=True,
        )

        df = pd.DataFrame({
            "Cay_Trong": labels,
            "Ty_Le_Phan_Tram": [f"{v:.1f}%" for v in values],
            "Do_Tu_Tin": [f"{c:.1f}%" for c in confs],
        })
        st.download_button(
            label="Tải kết quả phân tích",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"ket_qua_phan_tich_cay_trong_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_download_csv_file",
        )
