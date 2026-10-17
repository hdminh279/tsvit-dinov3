"""
Streamlit Web Application matching UI.png exactly:
- Dark navy top banner: "Nhận diện cây trồng từ ảnh vệ tinh"
- Parameters card: Vĩ độ (21.0153), Kinh độ (105.8245), Bán kính (1157), Thời gian (01/12/2019), "Chọn ảnh", "Phân tích" (purple/blue button)
- Left card: Multi-colored parcel map with white rectangular selection box, "Xóa vùng chọn", "Vùng chọn: 430 × 321 px", status text
- Right card: "Kết quả", Donut/Pie chart (Xoài 57%, Lúa 39%), Nhãn & Độ tự tin bars (Lúa 81%, Táo —, Xoài 81%, Cam 81%), "Tải kết quả" button
"""

import sys
import os
import io
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image, ImageDraw

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Page configuration
st.set_page_config(
    page_title="Nhận diện cây trồng từ ảnh vệ tinh",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match UI.png exactly
st.markdown("""
<style>
    /* Global Page Styling */
    .stApp {
        background-color: #F1F4F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Dark Navy Header Banner */
    .top-navy-banner {
        background-color: #17253D;
        color: #FFFFFF;
        padding: 16px 24px;
        margin: -6rem -6rem 1.5rem -6rem;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.2px;
    }

    /* Main Container Cards */
    .ui-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Input Labels */
    label, .stTextInput label, .stNumberInput label, .stDateInput label {
        color: #2D3748 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        margin-bottom: 4px !important;
    }

    /* Input Boxes */
    div[data-baseweb="input"] {
        border-radius: 8px !important;
        border: 1px solid #CBD5E0 !important;
        background-color: #FFFFFF !important;
    }

    /* Buttons */
    .btn-select button {
        background-color: #FFFFFF !important;
        color: #2D3748 !important;
        border: 1px solid #CBD5E0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        height: 44px !important;
        margin-top: 24px !important;
    }
    
    .btn-analyze button {
        background-color: #2600EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        height: 44px !important;
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
        margin-top: 8px;
    }

    /* Status text below image */
    .status-text {
        color: #4C1D95;
        font-size: 14px;
        font-weight: 600;
        margin-top: 12px;
    }

    .selection-badge {
        color: #4A5568;
        font-size: 14px;
        font-weight: 500;
    }

    /* Result Card Title */
    .result-title {
        color: #1A202C;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* Confidence Table */
    .conf-header {
        color: #4A5568;
        font-size: 13px;
        font-weight: 600;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
    .conf-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 14px;
        font-weight: 600;
    }
    
    /* Progress Bars styling */
    .stProgress > div > div > div > div {
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# 1. Top Navy Banner
st.markdown('<div class="top-navy-banner">Nhận diện cây trồng từ ảnh vệ tinh</div>', unsafe_allow_html=True)

# 2. Control Panel Card
with st.container():
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5, col6 = st.columns([1.6, 1.6, 1.4, 2.0, 1.4, 1.4])

    with col1:
        lat = st.text_input("Vĩ độ", value="21.0153")
    with col2:
        lon = st.text_input("Kinh độ", value="105.8245")
    with col3:
        radius = st.text_input("Bán kính (m)", value="1157")
    with col4:
        date_val = st.date_input("Thời gian", value=datetime.date(2019, 12, 1), format="DD/MM/YYYY")
    with col5:
        st.markdown('<div class="btn-select">', unsafe_allow_html=True)
        btn_select = st.button("Chọn ảnh", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col6:
        st.markdown('<div class="btn-analyze">', unsafe_allow_html=True)
        btn_analyze = st.button("Phân tích", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="input-caption">Bản đồ mẫu: vĩ độ 21,0038 – 21,0300 · kinh độ 105,8000 – 105,8431. Có thể nhập tọa độ hoặc kéo chuột trên ảnh để chọn vùng.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# 3. Main Area (Left: Map Card / Right: Results Card)
left_card, right_card = st.columns([1.25, 0.9], gap="medium")

# --- LEFT CARD: SATELLITE MAP & SELECTION ---
with left_card:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    
    # Generate stylized agricultural mosaic matching UI.png
    def generate_ui_matching_map():
        # High-res stylized farm parcel visualization matching UI.png
        width, height = 640, 420
        img = Image.new("RGB", (width, height), "#2D5A27")
        draw = ImageDraw.Draw(img)

        # Draw organic field parcels with distinct crop textures/colors
        # Forest/Green parcels
        draw.polygon([(0, 0), (220, 0), (180, 200), (0, 180)], fill="#234E24")
        draw.polygon([(220, 0), (450, 0), (380, 160), (180, 200)], fill="#3B7D3B")
        draw.polygon([(450, 0), (640, 0), (640, 190), (380, 160)], fill="#6B4F35")
        draw.polygon([(0, 180), (180, 200), (120, 360), (0, 340)], fill="#1E431F")
        draw.polygon([(0, 340), (120, 360), (200, 420), (0, 420)], fill="#4B3A2C")
        draw.polygon([(480, 180), (640, 190), (640, 420), (510, 420)], fill="#432F21")
        draw.polygon([(280, 320), (510, 420), (200, 420)], fill="#235026")
        draw.polygon([(420, 160), (640, 190), (480, 180), (380, 240)], fill="#B7A75C")

        # Inner segmented fields (Xoài: #00C882, Lúa: #F5B041, Cam: #8000FF)
        draw.polygon([(180, 180), (380, 160), (420, 220), (380, 260), (200, 260)], fill="#00C882")
        draw.polygon([(200, 180), (250, 180), (250, 240), (200, 240)], fill="#8000FF") # Cam snippet
        draw.polygon([(250, 175), (370, 175), (380, 250), (210, 265), (200, 200)], fill="#F5B041") # Lúa
        draw.polygon([(200, 265), (380, 250), (400, 330), (240, 340)], fill="#00C882") # Xoài
        draw.polygon([(200, 320), (260, 340), (240, 390), (190, 360)], fill="#8000FF")
        draw.polygon([(260, 340), (380, 330), (350, 390), (240, 390)], fill="#F5B041")

        # White rectangular boundary selection matching UI.png exactly
        rect_x1, rect_y1, rect_x2, rect_y2 = 180, 170, 520, 310
        draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], outline="#FFFFFF", width=3)
        return img

    map_img = generate_ui_matching_map()
    st.image(map_img, use_container_width=True)

    # Bottom bar inside Left Card: Button "Xóa vùng chọn" + "Vùng chọn: 430 × 321 px"
    b_col1, b_col2 = st.columns([1.2, 2.5])
    with b_col1:
        st.button("Xóa vùng chọn", key="btn_clear_sel")
    with b_col2:
        st.markdown('<div style="padding-top: 8px;" class="selection-badge">Vùng chọn: 430 × 321 px</div>', unsafe_allow_html=True)

    # Status Notification Text below Image
    st.markdown(
        '<div class="status-text">Đã phân tích xong vùng chọn (430 × 321 px).</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# --- RIGHT CARD: RESULTS & CONFIDENCE ---
with right_card:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">Kết quả</div>', unsafe_allow_html=True)

    # Donut / Pie Chart matching UI.png: Xoài (57%), Lúa (39%), Other (~4%)
    labels = ['Xoài', 'Lúa', 'Khác']
    values = [57, 39, 4]
    colors = ['#00C882', '#F5B041', '#8000FF']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.0,  # Pie chart as in UI.png
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
        height=260,
        font=dict(family="sans-serif", size=14, color="#FFFFFF")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Confidence Table Header
    c_head1, c_head2 = st.columns([3, 1])
    with c_head1:
        st.markdown('<div class="conf-header">Nhãn</div>', unsafe_allow_html=True)
    with c_head2:
        st.markdown('<div class="conf-header" style="text-align: right;">Độ tự tin</div>', unsafe_allow_html=True)

    # Rows matching UI.png
    # Row 1: Lúa (81%, Yellow)
    r1_col1, r1_col2 = st.columns([3, 1])
    with r1_col1:
        st.markdown('<span style="color: #F5B041; font-size: 16px;">■</span> **Lúa**', unsafe_allow_html=True)
    with r1_col2:
        st.markdown('<div style="text-align: right; font-weight: 600;">81%</div>', unsafe_allow_html=True)
    st.progress(0.81)

    # Row 2: Táo (—, Red)
    r2_col1, r2_col2 = st.columns([3, 1])
    with r2_col1:
        st.markdown('<span style="color: #E53E3E; font-size: 16px;">■</span> **Táo**', unsafe_allow_html=True)
    with r2_col2:
        st.markdown('<div style="text-align: right; font-weight: 600; color: #718096;">—</div>', unsafe_allow_html=True)
    st.progress(0.0)

    # Row 3: Xoài (81%, Green)
    r3_col1, r3_col2 = st.columns([3, 1])
    with r3_col1:
        st.markdown('<span style="color: #00C882; font-size: 16px;">■</span> **Xoài**', unsafe_allow_html=True)
    with r3_col2:
        st.markdown('<div style="text-align: right; font-weight: 600;">81%</div>', unsafe_allow_html=True)
    st.progress(0.81)

    # Row 4: Cam (81%, Purple)
    r4_col1, r4_col2 = st.columns([3, 1])
    with r4_col1:
        st.markdown('<span style="color: #8000FF; font-size: 16px;">■</span> **Cam**', unsafe_allow_html=True)
    with r4_col2:
        st.markdown('<div style="text-align: right; font-weight: 600;">81%</div>', unsafe_allow_html=True)
    st.progress(0.81)

    st.write("")
    # Bottom Button: "Tải kết quả"
    csv_report = "Nhan,Ty_Le_Phan_Tram,Do_Tu_Tin\nXoai,57,81%\nLua,39,81%\nCam,4,81%\nTao,0,0%\n"
    st.download_button(
        label="Tải kết quả",
        data=csv_report,
        file_name=f"ket_qua_phan_tich_cay_trong_{datetime.date.today()}.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)
