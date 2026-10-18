"""
Streamlit Web Application for Sentinel-2 Crop Classification & Segmentation
Layout: single-screen dashboard (no page scroll), fits any screen size.
- Top banner
- Controls card: Latitude, Longitude, Radius, Date, Analyze button
- Left card: Interactive satellite map (fills the remaining height of the screen)
- Right card: Responsive donut chart + crop list (area %, confidence) + CSV export
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
# CSS: height is computed from the viewport (dvh), so everything fits any screen
#   --row-h  = height of the main row (map card / result card)
#   Everything above the row: padding + banner + controls card + gaps ≈ 195px
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Toast at bottom-right so it never covers the Analyze button */
    [data-testid="stToastContainer"] {
        top: auto !important; bottom: 16px !important; right: 16px !important;
    }

    /* Hide default Streamlit chrome */
    header[data-testid="stHeader"], footer, #MainMenu,
    [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;
    }

    /* Lock the page: no scrolling */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section.main {
        height: 100dvh !important;
        max-height: 100dvh !important;
        overflow: hidden !important;
        background-color: #F8FAFC !important;
        color: #1E293B !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .block-container {
        padding: 0.4rem 0.9rem 0.7rem 0.9rem !important;
        max-width: 100% !important;
        height: 100dvh !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
    }
    .block-container > div[data-testid="stVerticalBlock"] {
        flex: 1 1 0 !important;
        min-height: 0 !important;
    }

    div[data-testid="stVerticalBlock"] { gap: 0.4rem !important; }
    /* The <style> block itself is an empty element that still adds a gap: hide it */
    .block-container [data-testid="stElementContainer"]:has(style) { display: none !important; }
    .block-container > div[data-testid="stVerticalBlock"] > *:not(:has(.st-key-map_card)) {
        flex: 0 0 auto !important;
    }

    /* ---------- Banner ---------- */
    .top-navy-banner {
        background-color: #17253D;
        color: #FFFFFF !important;
        height: 38px;
        padding: 0 16px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: -0.2px;
        display: flex;
        align-items: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }

    /* ---------- Generic card (keyed containers) ---------- */
    .st-key-controls_card, .st-key-map_card, .st-key-result_card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.5rem 0.9rem !important;
        box-sizing: border-box;
    }
    .st-key-controls_card {
        gap: 0.1rem !important;
        padding-top: 0.45rem !important;
        padding-bottom: 0.5rem !important;
        flex: 0 0 auto !important;
        min-height: fit-content !important;
    }
    :is([data-testid="stElementContainer"], [data-testid="stLayoutWrapper"]):has(.top-navy-banner),
    :is([data-testid="stElementContainer"], [data-testid="stLayoutWrapper"]):has(.st-key-controls_card) {
        flex: 0 0 auto !important;
    }
    .top-navy-banner { flex: 0 0 auto !important; }
    :is([data-testid="stElementContainer"], [data-testid="stLayoutWrapper"]):has(.top-navy-banner) {
        height: 38px !important; min-height: 38px !important; flex: 0 0 38px !important;
    }
    .st-key-controls_card { position: relative; z-index: 1; }

    /* Controls: compact inputs, button aligned with inputs */
    .st-key-controls_card label p { font-size: 12.5px !important; }
    .st-key-controls_card div[data-testid="stNumberInput"],
    .st-key-controls_card div[data-testid="stDateInput"] { margin-bottom: 0 !important; }
    .st-key-controls_card div.stButton { margin-top: 18px !important; }
    .st-key-controls_card div.stButton > button {
        height: 32px !important; min-height: 32px !important; padding-top: 0 !important; padding-bottom: 0 !important;
    }
    /* Labels right on top of their inputs, inputs shorter */
    .st-key-controls_card [data-testid="stWidgetLabel"] {
        min-height: 0 !important; margin-bottom: 1px !important; padding: 0 !important;
    }
    .st-key-controls_card [data-testid="stWidgetLabel"] p { line-height: 1.1 !important; margin: 0 !important; }
    .st-key-controls_card [data-baseweb="input"],
    .st-key-controls_card [data-baseweb="base-input"],
    .st-key-controls_card div[data-testid="stDateInputField"] { min-height: 32px !important; height: 32px !important; }
    .st-key-controls_card input { height: 32px !important; padding-top: 0 !important; padding-bottom: 0 !important; }
    .st-key-controls_card [data-testid="stNumberInput"] button { height: 32px !important; }
    .st-key-controls_card div[data-testid="stVerticalBlock"],
    .st-key-controls_card div[data-testid="stElementContainer"] { gap: 0.1rem !important; }
    .st-key-controls_card [data-testid="stHorizontalBlock"] { gap: 0.75rem !important; }

    .input-caption {
        font-size: 11.5px !important;
        color: #64748B;
        line-height: 1.2 !important;
        margin: 0 !important;
    }

    /* ---------- Flex chain: every ancestor of the two cards fills the leftover height ---------- */
    .block-container :is([data-testid="stVerticalBlock"], [data-testid="stLayoutWrapper"],
                         [data-testid="stElementContainer"], [data-testid="stVerticalBlockBorderWrapper"]):has(.st-key-map_card, .st-key-result_card) {
        flex: 1 1 0 !important;
        min-height: 0 !important;
        height: auto !important;
        display: flex !important;
        flex-direction: column !important;
    }
    .block-container [data-testid="stHorizontalBlock"]:has(.st-key-map_card, .st-key-result_card) {
        flex: 1 1 0 !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: stretch !important;
    }
    .block-container :is([data-testid="stColumn"], [data-testid="column"]):has(.st-key-map_card, .st-key-result_card) {
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* The two cards */
    .st-key-map_card, .st-key-result_card {
        flex: 1 1 0 !important;
        min-height: 300px !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
    }

    /* ---------- Map fills whatever is left inside its card ---------- */
    .st-key-map_card > * { flex-shrink: 0; }
    .st-key-map_card [data-testid="stElementContainer"]:has(iframe) {
        flex: 1 1 0 !important;
        min-height: 0 !important;
        height: auto !important;
        margin-bottom: 60px !important;
        position: relative !important;
        overflow: hidden !important;
        border-radius: 8px !important;
    }
    .st-key-map_card [data-testid="stElementContainer"]:has(iframe) div:has(iframe),
    .st-key-map_card iframe {
        position: absolute !important;
        inset: 0 !important;
        height: 100% !important;
        width: 100% !important;
        min-height: 0 !important;
        border: none !important;
    }
    .map-title { font-size: 13.5px; font-weight: 600; color: #1E293B; padding-top: 4px; }
    .st-key-map_card { padding-bottom: 1rem !important; }
    .st-key-map_card .input-caption { padding: 2px 0 8px 0 !important; }

    /* ---------- Result card ---------- */
    .st-key-result_card [data-testid="stElementContainer"]:has(.result-wrap) {
        flex: 1 1 0 !important;
        min-height: 0 !important;
    }
    .st-key-result_card [data-testid="stElementContainer"]:has(.result-wrap) div:has(.result-wrap) {
        height: 100% !important;
    }
    .result-wrap {
        height: 100%;
        display: flex;
        flex-direction: column;
        gap: 6px;
        min-height: 0;
    }
    .result-title { font-size: 14px; font-weight: 700; color: #0F172A; flex: 0 0 auto; }

    /* Donut takes ONLY the leftover space, so it can never overlap the list */
    .donut-area {
        flex: 1 1 0;
        min-height: 70px;
        container-type: size;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .donut {
        width: min(100cqw, 100cqh, 200px);
        aspect-ratio: 1;
        border-radius: 50%;
        position: relative;
        border: 2px solid #FFFFFF;
        box-sizing: border-box;
        box-shadow: 0 1px 4px rgba(0,0,0,0.18);
    }
    .pie-label {
        position: absolute;
        transform: translate(-50%, -50%);
        z-index: 2;
        color: #FFFFFF;
        text-align: center;
        line-height: 1.15;
        font-size: clamp(8px, 1.35dvh, 11px);
        font-weight: 600;
        text-shadow: 0 1px 2px rgba(0,0,0,0.45);
        pointer-events: none;
        white-space: nowrap;
    }
    .pie-label b { display: block; font-size: clamp(9px, 1.6dvh, 13px); font-weight: 700; }

    /* Crop list: natural height, compact rows */
    .crop-head {
        flex: 0 0 auto;
        display: grid;
        grid-template-columns: 1fr 62px;
        font-size: 11.5px;
        font-weight: 600;
        color: #475569;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 3px;
    }
    .crop-head span:nth-child(n+2), .crop-top span:nth-child(n+2) { text-align: right; }
    .crop-list {
        flex: 0 0 auto;
        display: flex;
        flex-direction: column;
        gap: 5px;
        padding-bottom: 6px;
    }
    .crop-row { display: flex; flex-direction: column; gap: 3px; }
    .crop-top {
        display: grid;
        grid-template-columns: 1fr 62px;
        align-items: center;
        font-size: 12.5px;
        line-height: 1.2;
    }
    .crop-name { font-weight: 600; color: #0F172A; display: flex; align-items: center; gap: 6px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
    .dot { width: 10px; height: 10px; border-radius: 3px; display: inline-block; flex: 0 0 auto; }
    .crop-conf { font-weight: 700; color: #1E293B; }
    .bar { height: 5px; background: #EEF2F7; border-radius: 4px; overflow: hidden; }
    .bar > div { height: 100%; border-radius: 4px; }

    .st-key-result_card div.stDownloadButton { flex: 0 0 auto; margin-top: 8px !important; margin-bottom: 4px !important; }
    .st-key-result_card div.stDownloadButton > button {
        height: 30px !important; min-height: 30px !important;
        font-size: 12px !important; padding: 0 0.75rem !important;
    }

    ::-webkit-scrollbar { display: none !important; width: 0 !important; height: 0 !important; }
    * { scrollbar-width: none !important; -ms-overflow-style: none !important; }

    /* ---------- Fallback: very small screens / phones -> allow scrolling ---------- */
    @media (max-height: 560px), (max-width: 768px) {
        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stMain"], section.main, .block-container {
            height: auto !important;
            max-height: none !important;
            overflow: auto !important;
        }
        .block-container { display: block !important; }
        .st-key-map_card, .st-key-result_card { height: 520px !important; flex: none !important; }
    }
</style>
""", unsafe_allow_html=True)


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
    """Builds the responsive pie chart (CSS conic-gradient) + crop list as one-line HTML."""
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
        if share >= 5:  # label only slices big enough to hold text
            ang = (start + end) / 2 * 2 * math.pi
            r = 0.30 if share < 15 else 0.28
            x = 50 + r * 100 * math.sin(ang)
            y = 50 - r * 100 * math.cos(ang)
            name_html = f"{name}<br>" if share >= 10 else ""
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
        '<div class="result-title">Kết quả phân tích</div>'
        '<div class="donut-area">'
        f'<div class="donut" style="background: conic-gradient({gradient});">{pie_labels}</div></div>'
        '<div class="crop-head"><span>Cây trồng</span><span>Độ tự tin</span></div>'
        f'<div class="crop-list">{rows}</div>'
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
# Banner
# ---------------------------------------------------------------------------
st.markdown('<div class="top-navy-banner">Nhận diện cây trồng từ ảnh vệ tinh</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Controls card
# ---------------------------------------------------------------------------
with st.container(key="controls_card"):
    c1, c2, c3, c4, c5 = st.columns([1.5, 1.5, 1.3, 1.5, 1.4])
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

# Run analysis only on button click or cold start
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
# Main row
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1.7, 1], gap="small")

# --- LEFT: MAP -------------------------------------------------------------
with col_left:
    with st.container(key="map_card"):
        m_head1, m_head2 = st.columns([2, 1.5])
        with m_head1:
            st.markdown('<div class="map-title">Bản đồ vệ tinh tương tác</div>', unsafe_allow_html=True)
        with m_head2:
            show_layer = st.checkbox("Hiển thị lớp nhãn màu", value=True, key="cb_show_overlay")

        m = folium.Map(
            location=[st.session_state.lat, st.session_state.lon],
            zoom_start=14,
            tiles=None,
            control_scale=True,
        )

        m.get_root().header.add_child(folium.Element(
            "<style>html,body{height:100%;margin:0;padding:0;}"
            ".folium-map{width:100% !important;height:100% !important;}</style>"
        ))

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

        # Real height is controlled by CSS (flex fill); this value is only a fallback.
        map_data = st_folium(
            m,
            height=500,
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
            '<div class="input-caption">💡 <b>Mẹo:</b> Nhấp chuột bất kỳ đâu trên bản đồ để di chuyển ô vuông, '
            'sau đó bấm <b>Phân tích</b>. Có thể phóng to/thu nhỏ tự do.</div>',
            unsafe_allow_html=True,
        )

# --- RIGHT: RESULTS --------------------------------------------------------
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

        st.markdown(build_result_html(labels, values, confs, colors), unsafe_allow_html=True)

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
