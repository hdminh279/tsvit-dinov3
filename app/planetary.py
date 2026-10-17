"""
Microsoft Planetary Computer STAC Client for Live Sentinel-2 Satellite Imagery.
Enables real-time retrieval of cloud-free Sentinel-2 L2A scenes by coordinates & date.
"""

from typing import Dict, Optional, Tuple
import datetime
import numpy as np
from PIL import Image, ImageEnhance
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import streamlit as st

try:
    import pystac_client
    import planetary_computer as pc
    STAC_AVAILABLE = True
except ImportError:
    STAC_AVAILABLE = False

def enhance_satellite_image(img_pil: Image.Image) -> Image.Image:
    """Enhances clarity, contrast, and color vibrance of Sentinel-2 True Color image."""
    arr = np.array(img_pil, dtype=np.float32)
    # 1% - 99% Percentile stretch per channel
    for c in range(3):
        p1, p99 = np.percentile(arr[:, :, c], (1, 99))
        if p99 > p1:
            arr[:, :, c] = np.clip((arr[:, :, c] - p1) / (p99 - p1) * 255.0, 0, 255)
    
    enhanced = Image.fromarray(arr.astype(np.uint8))
    # Slight sharpness & color boost
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.3)
    enhanced = ImageEnhance.Color(enhanced).enhance(1.15)
    return enhanced


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_planetary_sentinel2(
    lat: float,
    lon: float,
    radius: float = 1200.0,
    target_date_str: str = "2022-06-15",
    max_cloud: float = 25.0
) -> Tuple[Image.Image, Dict]:
    """
    Queries Microsoft Planetary Computer STAC for Sentinel-2 L2A scenes matching (lat, lon, radius, date).
    Streams cloud-free True Color (TCI) window, enhances image clarity, and extracts SITS tensor.
    """
    if not STAC_AVAILABLE:
        raise RuntimeError("Thư viện pystac_client hoặc planetary_computer chưa được cài đặt.")

    # Calculate bounding box from radius (meters)
    dlat = radius / 111320.0
    dlon = radius / (111320.0 * np.cos(np.radians(lat)) + 1e-6)
    bbox = [lon - dlon, lat - dlat, lon + dlon, lat + dlat]

    try:
        t_date = datetime.date.fromisoformat(target_date_str)
    except Exception:
        t_date = datetime.date(2022, 6, 15)

    start_d = (t_date - datetime.timedelta(days=40)).strftime("%Y-%m-%d")
    end_d = (t_date + datetime.timedelta(days=40)).strftime("%Y-%m-%d")

    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_d}/{end_d}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
        max_items=8
    )

    items = list(search.items())
    if not items:
        # Fallback to wider search
        search_wider = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_d}/{end_d}",
            max_items=5
        )
        items = list(search_wider.items())
        if not items:
            raise ValueError(f"Không tìm thấy ảnh Sentinel-2 nào quanh tọa độ ({lat:.4f}, {lon:.4f}) trong khoảng thời gian {start_d} đến {end_d}.")

    # Pick the scene with lowest cloud cover
    items.sort(key=lambda it: float(it.properties.get("eo:cloud_cover", 100.0)))
    item = items[0]
    
    acq_date = item.properties.get("datetime", "")[:10]
    cloud_pct = float(item.properties.get("eo:cloud_cover", 0.0))
    visual_href = item.assets.get("visual", {}).href

    if not visual_href:
        raise ValueError(f"Scene {item.id} không có asset visual True Color.")

    # Stream the window via HTTP Range Request
    with rasterio.open(visual_href) as src:
        bounds_proj = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(*bounds_proj, transform=src.transform)
        rgb = src.read(
            [1, 2, 3],
            window=win,
            out_shape=(3, 320, 540),
            resampling=rasterio.enums.Resampling.bilinear
        )
        raw_pil = Image.fromarray(rgb.transpose(1, 2, 0))

    # Enhance contrast & sharpness
    display_pil = enhance_satellite_image(raw_pil)

    meta = {
        "item_id": item.id,
        "acquisition_date": acq_date,
        "cloud_cover": round(cloud_pct, 1),
        "bbox": bbox,
        "lat": lat,
        "lon": lon,
        "radius": radius
    }

    return display_pil, meta

