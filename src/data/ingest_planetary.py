"""
Sentinel-2 ingestion via Microsoft Planetary Computer STAC API (Fallback).
Enables free cloud-optimized GeoTIFF (COG) retrieval when GEE authentication is unavailable.
"""

from typing import Dict, List, Optional, Tuple
import datetime
import numpy as np

try:
    from pystac_client import Client
    import planetary_computer as pc
    STAC_AVAILABLE = True
except ImportError:
    STAC_AVAILABLE = False

from .constants import S2_10BANDS


class PlanetarySentinel2Ingestor:
    """
    STAC-based ingestion for Sentinel-2 L2A from Microsoft Planetary Computer.
    """
    STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    COLLECTION_ID = "sentinel-2-l2a"
    
    # Map S2_10BANDS to Planetary Computer asset keys
    ASSET_MAP = {
        'B02': 'B02', 'B03': 'B03', 'B04': 'B04',
        'B05': 'B05', 'B06': 'B06', 'B07': 'B07',
        'B08': 'B08', 'B8A': 'B8A', 'B11': 'B11', 'B12': 'B12'
    }

    def __init__(self):
        self.client = None
        if STAC_AVAILABLE:
            try:
                self.client = Client.open(self.STAC_API_URL, modifier=pc.sign_inplace)
            except Exception as e:
                print(f"[STAC] Warning opening Planetary Computer STAC client: {e}")

    def search_items(
        self,
        bbox: List[float],  # [min_lon, min_lat, max_lon, max_lat]
        start_date: str = '2019-01-01',
        end_date: str = '2019-12-01',
        max_cloud_cover: float = 20.0
    ):
        """Searches Sentinel-2 L2A scenes matching bounding box, date range, and cloud threshold."""
        if not self.client:
            raise RuntimeError("pystac_client or planetary_computer is not available.")

        date_range = f"{start_date}/{end_date}"
        search = self.client.search(
            collections=[self.COLLECTION_ID],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            sortby=[{"field": "properties.datetime", "direction": "asc"}]
        )
        items = list(search.items())
        return items

    def extract_time_series_metadata(self, items) -> Tuple[List[str], List[int]]:
        """Extract item IDs and Day-of-Year values."""
        item_ids = []
        doys = []
        for item in items:
            dt_str = item.properties.get("datetime")
            if dt_str:
                dt = datetime.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                doys.append(dt.timetuple().tm_yday)
                item_ids.append(item.id)
        return item_ids, doys
