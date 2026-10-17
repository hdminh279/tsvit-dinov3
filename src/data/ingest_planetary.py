"""
Sentinel-2 ingestion via Microsoft Planetary Computer STAC API (Fallback).
Enables free cloud-optimized GeoTIFF (COG) retrieval without Earth Engine credentials.
"""

from typing import Dict, List, Optional, Tuple
import os
import json
import argparse
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

    def extract_time_series_metadata(self, items) -> List[Dict]:
        """Extract scene metadata, IDs, DOY, and signed asset URLs for the 10 bands."""
        records = []
        for item in items:
            dt_str = item.properties.get("datetime")
            doy = None
            if dt_str:
                dt = datetime.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                doy = dt.timetuple().tm_yday

            assets_urls = {}
            for band in S2_10BANDS:
                asset_key = self.ASSET_MAP.get(band, band)
                if asset_key in item.assets:
                    assets_urls[band] = item.assets[asset_key].href

            records.append({
                "id": item.id,
                "datetime": dt_str,
                "date": dt_str[:10] if dt_str else None,
                "doy": doy,
                "cloud_cover": float(item.properties.get("eo:cloud_cover", 0.0)),
                "platform": item.properties.get("platform", "Sentinel-2"),
                "assets": assets_urls
            })
        return records


def main():
    parser = argparse.ArgumentParser(description="Ingest Sentinel-2 L2A time-series from Microsoft Planetary Computer")
    parser.add_argument("--bbox", nargs=4, type=float, required=True,
                        help="Bounding box: min_lon min_lat max_lon max_lat (e.g. 1.95 47.95 2.05 48.05)")
    parser.add_argument("--start_date", type=str, default="2019-01-01",
                        help="Start acquisition date YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, default="2019-12-01",
                        help="End acquisition date YYYY-MM-DD")
    parser.add_argument("--cloud_cover_max", type=float, default=20.0,
                        help="Maximum cloudy pixel percentage (default: 20)")
    parser.add_argument("--output_dir", type=str, default="data/raw/planetary",
                        help="Directory to save metadata and time-series records")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Planetary] Connecting to Microsoft Planetary Computer STAC API...")
    print(f"[Planetary] Search criteria: BBOX={args.bbox}, Dates={args.start_date} to {args.end_date}, MaxCloud={args.cloud_cover_max}%")

    ingestor = PlanetarySentinel2Ingestor()
    items = ingestor.search_items(
        bbox=args.bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        max_cloud_cover=args.cloud_cover_max
    )

    print(f"[Planetary] Found {len(items)} matching Sentinel-2 L2A scenes.")
    metadata = ingestor.extract_time_series_metadata(items)

    out_file = os.path.join(args.output_dir, "scenes_metadata.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Planetary] Saved scene catalog to: {out_file}")
    print("\n--- Danh sách các mốc thời gian (Timestamps) thu thập được ---")
    for idx, sc in enumerate(metadata):
        print(f"[{idx+1:02d}] Ngày: {sc['date']} | DOY: {sc['doy']:03d} | Mây: {sc['cloud_cover']:5.1f}% | ID: {sc['id']}")

    print(f"\n[Planetary] Hoàn tất! Đã thu thập siêu dữ liệu của {len(metadata)} mốc ảnh Sentinel-2 sẵn sàng xử lý.")


if __name__ == "__main__":
    main()
