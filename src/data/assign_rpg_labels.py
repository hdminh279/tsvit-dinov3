"""
Assign and update RPG-2022 ground truth agricultural crop labels to existing .pt patches.
Decouples label assignment from imagery downloading to save hours of network bandwidth.
"""

from typing import Optional, Tuple
import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds

from .constants import CROP_PATCH_SIZE, CROP_CLASSES_EN
from .rpg_processor import RPGProcessor


def update_single_patch(item: Tuple[str, int, int, np.ndarray]) -> Tuple[str, int]:
    """Updates label tensor in a single .pt patch file."""
    file_path, y, x, label_raster = item
    try:
        sample = torch.load(file_path, weights_only=False)
        patch_lbl = np.ascontiguousarray(label_raster[y:y + CROP_PATCH_SIZE, x:x + CROP_PATCH_SIZE])
        sample['label'] = torch.from_numpy(patch_lbl).long()
        torch.save(sample, file_path)
        valid_count = int(np.count_nonzero(patch_lbl > 0))
        return file_path, valid_count
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return file_path, 0


def assign_labels(
    patches_dir: str = "data/processed/patches",
    rpg_vector_file: str = "data/raw/rpg_2022/PARCELLES_GRAPHIQUES.shp",
    metadata_json: Optional[str] = None,
    bbox: Tuple[float, float, float, float] = (1.35, 47.88, 2.27, 48.54),
    num_workers: int = 8
):
    """
    Loads RPG vector parcels, rasterizes full AOI grid, and assigns real crop labels to .pt patches.
    """
    if not os.path.exists(rpg_vector_file) or os.path.isdir(rpg_vector_file):
        search_dir = rpg_vector_file if os.path.isdir(rpg_vector_file) else os.path.dirname(rpg_vector_file)
        if not os.path.exists(search_dir):
            search_dir = "data/raw/rpg_2022" if os.path.exists("data/raw/rpg_2022") else "data/raw"
        found = None
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.endswith(".shp") and ("PARCELLES" in f or "RPG" in f):
                    found = os.path.join(root, f)
                    break
            if found:
                break
        if found:
            print(f"[Labeler] Auto-discovered RPG shapefile at: {found}")
            rpg_vector_file = found
        else:
            raise FileNotFoundError(
                f"RPG vector file not found at: {rpg_vector_file}\n"
                "Please download and extract RPG 2022 first (see instructions)."
            )

    meta_file = metadata_json or os.path.join(patches_dir, "metadata.json")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Patches metadata not found at: {meta_file}")

    with open(meta_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        print(f"[Labeler] No patch records found in {meta_file}")
        return

    # Infer spatial window from patch bounds
    max_y = max(r["y"] for r in records) + CROP_PATCH_SIZE
    max_x = max(r["x"] for r in records) + CROP_PATCH_SIZE
    win_h, win_w = int(max_y), int(max_x)

    print(f"[Labeler] Target AOI dimensions: {win_w} x {win_h} pixels")
    print(f"[Labeler] Loading agricultural parcels from: {rpg_vector_file}...")
    processor = RPGProcessor()
    parcels = RPGProcessor.load_parcels_from_file(rpg_vector_file, bounds=bbox)
    print(f"[Labeler] Extracted {len(parcels)} parcel polygons inside AOI.")

    print(f"[Labeler] Rasterizing parcels to 10m spatial grid ({win_h} x {win_w})...")
    label_raster = processor.rasterize_parcels(parcels, bounds=bbox, out_shape=(win_h, win_w))

    # Calculate class distribution
    unique, counts = np.unique(label_raster, return_counts=True)
    dist = dict(zip(unique, counts))
    print("\n--- Crop Class Distribution in AOI ---")
    for cls_idx in sorted(dist.keys()):
        cls_name = CROP_CLASSES_EN.get(int(cls_idx), "Unknown")
        pct = dist[cls_idx] / label_raster.size * 100
        print(f"  Class {cls_idx:02d} ({cls_name:25s}): {dist[cls_idx]:8d} pixels ({pct:5.2f}%)")

    # Update patches concurrently
    tasks = []
    for r in records:
        f_path = os.path.join(patches_dir, r["file"])
        if os.path.exists(f_path):
            tasks.append((f_path, r["y"], r["x"], label_raster))

    print(f"\n[Labeler] Updating labels in {len(tasks)} .pt patches using {num_workers} threads...")
    total_valid_pixels = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for _, valid_count in executor.map(update_single_patch, tasks):
            total_valid_pixels += valid_count

    print(f"[Labeler] Completed! Successfully updated {len(tasks)} patches.")
    print(f"[Labeler] Total labeled cropland pixels assigned: {total_valid_pixels}")


def main():
    parser = argparse.ArgumentParser(description="Assign RPG 2022 crop labels to pre-extracted .pt patches")
    parser.add_argument("--patches_dir", type=str, default="data/processed/patches",
                        help="Directory containing patch_*.pt files")
    parser.add_argument("--rpg_file", type=str, default="data/raw/rpg_2022/PARCELLES_GRAPHIQUES.shp",
                        help="Path to real RPG 2022 shapefile")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Path to patches metadata.json")
    parser.add_argument("--bbox", nargs=4, type=float, default=[1.35, 47.88, 2.27, 48.54],
                        help="AOI BBOX: min_lon min_lat max_lon max_lat (default: 60k patches)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker threads")

    args = parser.parse_args()
    assign_labels(
        patches_dir=args.patches_dir,
        rpg_vector_file=args.rpg_file,
        metadata_json=args.metadata,
        bbox=tuple(args.bbox),
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()
