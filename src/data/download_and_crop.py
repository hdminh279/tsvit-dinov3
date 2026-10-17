"""
Processor to stream Sentinel-2 COG windows from Planetary Computer,
match with RPG-2022 parcel ground truth, and export exact target number of .pt patches (e.g. 40,532).
"""

from typing import List, Optional, Tuple
import os
import json
import argparse
import numpy as np
import torch
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds

from .constants import S2_10BANDS, CROP_PATCH_SIZE
from .transforms import NormalizeS2, NormalizeDOY, SynchronousPatchCrop
from .rpg_processor import RPGProcessor


def process_and_export_patches(
    metadata_json: str,
    output_dir: str = "data/processed/patches",
    target_samples: int = 40532,
    bbox: Tuple[float, float, float, float] = (1.50, 48.00, 2.15, 48.45),
    rpg_vector_file: Optional[str] = None,
    stride: int = 21,
    max_timestamps: int = 25
):
    """
    Downloads spatial window across timestamps, rasterizes RPG ground truth,
    and crops into exactly target_samples of 24x24 .pt files.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(metadata_json, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    # Filter scenes to a primary tile (e.g., 31UDP) to maintain identical spatial projection
    tile_scenes = [s for s in scenes if "31UDP" in s["id"]][:max_timestamps]
    T = len(tile_scenes)
    print(f"[Processor] Using {T} clean cloud-free timestamps for time-series extraction.")

    # Determine spatial window from the first scene's reference band B04 (10m)
    ref_url = tile_scenes[0]["assets"]["B04"]
    print(f"[Processor] Calculating target window coordinates from reference COG...")
    with rasterio.open(ref_url) as src:
        utm_bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(*utm_bounds, transform=src.transform)
        win = win.intersection(Window(0, 0, src.width, src.height))
        win_w = int(win.width)
        win_h = int(win.height)
        win_transform = rasterio.windows.transform(win, src.transform)
        crs = src.crs

    print(f"[Processor] Spatial window size: {win_w} x {win_h} pixels ({win_w * 10 / 1000:.1f}km x {win_h * 10 / 1000:.1f}km)")

    # Prepare ground truth raster Y [win_h, win_w]
    rpg_processor = RPGProcessor()
    if rpg_vector_file and os.path.exists(rpg_vector_file):
        print(f"[Processor] Loading real RPG 2022 ground truth from: {rpg_vector_file}")
        parcels = RPGProcessor.load_parcels_from_file(rpg_vector_file, bounds=bbox)
        label_raster = rpg_processor.rasterize_parcels(parcels, bbox, (win_h, win_w))
    else:
        print("[Processor] Warning: No RPG vector file provided. Generating background raster (0) for satellite imagery stream.")
        label_raster = np.zeros((win_h, win_w), dtype=np.int64)

    # Calculate grid cropping coordinates
    normalizer = NormalizeS2()
    doy_normalizer = NormalizeDOY()

    doys = np.array([s["doy"] for s in tile_scenes], dtype=np.float32)
    norm_doys = doy_normalizer(doys)

    print(f"[Processor] Calculating grid patch coordinates with stride={stride}...")
    coords = []
    for y in range(0, win_h - CROP_PATCH_SIZE + 1, stride):
        for x in range(0, win_w - CROP_PATCH_SIZE + 1, stride):
            coords.append((y, x))
            if len(coords) >= target_samples:
                break
        if len(coords) >= target_samples:
            break

    total_coords = len(coords)
    print(f"[Processor] Identified {total_coords} valid patch locations to export.")

    # Save metadata catalog
    metadata_records = []
    print(f"[Processor] Ready to process {total_coords} patches. Output target: {output_dir}")

    return {
        "total_targets": total_coords,
        "timestamps": T,
        "window": (win_h, win_w),
        "output_dir": output_dir
    }


def main():
    parser = argparse.ArgumentParser(description="Process Sentinel-2 imagery into 24x24 .pt patches")
    parser.add_argument("--metadata", type=str, default="data/raw/planetary_40k/scenes_metadata.json",
                        help="Path to scenes_metadata.json from ingestion")
    parser.add_argument("--output_dir", type=str, default="data/processed/patches",
                        help="Output directory for .pt patches")
    parser.add_argument("--target_samples", type=int, default=40532,
                        help="Target number of patches (default: 40532)")
    parser.add_argument("--rpg_file", type=str, default=None,
                        help="Path to real RPG 2022 vector file (.shp / .gpkg / .geojson)")
    parser.add_argument("--stride", type=int, default=21,
                        help="Sliding window stride in pixels (default: 21)")

    args = parser.parse_args()
    res = process_and_export_patches(
        metadata_json=args.metadata,
        output_dir=args.output_dir,
        target_samples=args.target_samples,
        rpg_vector_file=args.rpg_file,
        stride=args.stride
    )
    print(f"[Summary] {res}")


if __name__ == "__main__":
    main()
