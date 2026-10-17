"""
Sentinel-2 COG patch extractor and processor.
Streams multi-temporal imagery from Microsoft Planetary Computer,
aligns with RPG-2022 ground truth parcels, and serializes 24x24 .pt patches.
Includes auto-resume support, token caching, and GDAL socket timeout protection.
"""

from typing import Optional, Tuple
import os
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
import planetary_computer as pc

from .constants import S2_10BANDS, CROP_PATCH_SIZE
from .transforms import NormalizeS2, NormalizeDOY
from .rpg_processor import RPGProcessor

import threading

# Container-level SAS token manager (valid for 1 hour, auto-refreshed thread-safely)
_CONTAINER_TOKEN = None
_CONTAINER_EXPIRY = 0
_TOKEN_LOCK = threading.Lock()


def get_container_sas_token() -> str:
    """Gets and caches a 1-hour container-level SAS token for Sentinel-2 blob storage."""
    global _CONTAINER_TOKEN, _CONTAINER_EXPIRY
    now = time.time()
    if _CONTAINER_TOKEN and (now < _CONTAINER_EXPIRY - 180):
        return _CONTAINER_TOKEN
    with _TOKEN_LOCK:
        if _CONTAINER_TOKEN and (time.time() < _CONTAINER_EXPIRY - 180):
            return _CONTAINER_TOKEN
        for attempt in range(5):
            try:
                tok = pc.sas.get_token('sentinel2l2a01', 'sentinel2-l2')
                _CONTAINER_TOKEN = tok.token
                _CONTAINER_EXPIRY = tok.expiry.timestamp() if hasattr(tok.expiry, 'timestamp') else (time.time() + 3600)
                return _CONTAINER_TOKEN
            except Exception as e:
                time.sleep(1 + attempt)
        return _CONTAINER_TOKEN or ""


def get_fresh_url(url: str) -> str:
    """Appends fresh container SAS token to unsigned COG url."""
    base_url = url.split("?")[0]
    token = get_container_sas_token()
    return f"{base_url}?{token}" if token else url


def fetch_single_band(item: Tuple) -> Tuple[int, int, np.ndarray, Optional[str]]:
    """Worker function to stream a single band via HTTP Range Request with timeout protection and auto-retry."""
    t_idx, b_idx, raw_url, strip_bounds, strip_h, win_w = item
    last_err = None
    for attempt in range(3):
        url = get_fresh_url(raw_url)
        try:
            with rasterio.open(url) as src:
                band_win = from_bounds(*strip_bounds, transform=src.transform)
                arr = src.read(
                    1,
                    window=band_win,
                    out_shape=(strip_h, win_w),
                    resampling=Resampling.bilinear
                )
                return t_idx, b_idx, arr, None
        except Exception as e:
            last_err = str(e)
            time.sleep(1 + attempt)
    return t_idx, b_idx, np.zeros((strip_h, win_w), dtype=np.float32), last_err


def process_and_export_patches(
    metadata_json: str,
    output_dir: str = "data/processed/patches",
    target_samples: Optional[int] = None,
    bbox: Tuple[float, float, float, float] = (1.35, 47.88, 2.27, 48.54),
    rpg_vector_file: Optional[str] = None,
    stride: int = 24,
    max_timestamps: int = 25,
    strip_height: int = 252,
    num_workers: int = 8
) -> dict:
    """
    Streams multi-temporal Sentinel-2 imagery in horizontal strips,
    aligns with RPG parcel labels, and saves 24x24 .pt patches.
    Supports auto-resume from existing files to prevent repeating work.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(metadata_json, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    # Filter to primary tile (31UDP) to preserve unified CRS
    tile_scenes = [s for s in scenes if "31UDP" in s.get("id", "")][:max_timestamps]
    T = len(tile_scenes)
    print(f"[Extractor] Selected {T} clean cloud-free timestamps.")

    # Calculate spatial window from reference band B04 (10m)
    ref_url = get_fresh_url(tile_scenes[0]["assets"]["B04"])
    print("[Extractor] Querying spatial grid from reference COG...")
    gdal_env_opts = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_TIMEOUT": "15",
        "GDAL_HTTP_CONNECTTIMEOUT": "5",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "1",
        "VSI_CACHE": "YES"
    }

    with rasterio.Env(**gdal_env_opts):
        with rasterio.open(ref_url) as src:
            utm_bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
            win = from_bounds(*utm_bounds, transform=src.transform)
            win = win.intersection(Window(0, 0, src.width, src.height))
            win_w = int(win.width)
            win_h = int(win.height)
            win_col_off = int(win.col_off)
            win_row_off = int(win.row_off)
            ref_transform = src.transform

    print(f"[Extractor] AOI Grid: {win_w} x {win_h} pixels ({win_w * 0.01:.1f} km x {win_h * 0.01:.1f} km)")

    # Prepare ground truth raster
    rpg_processor = RPGProcessor()
    if rpg_vector_file and os.path.exists(rpg_vector_file):
        print(f"[Extractor] Rasterizing RPG 2022 labels from: {rpg_vector_file}")
        parcels = RPGProcessor.load_parcels_from_file(rpg_vector_file, bounds=bbox)
        label_raster = rpg_processor.rasterize_parcels(parcels, bbox, (win_h, win_w))
    else:
        print("[Extractor] Notice: No RPG file provided. Defaulting to background class (0).")
        label_raster = np.zeros((win_h, win_w), dtype=np.int64)

    normalizer = NormalizeS2()
    doy_normalizer = NormalizeDOY()
    doys = np.array([s["doy"] for s in tile_scenes], dtype=np.float32)
    norm_doys = doy_normalizer(doys)

    total_y_steps = len(range(0, win_h - CROP_PATCH_SIZE + 1, stride))
    total_x_steps = len(range(0, win_w - CROP_PATCH_SIZE + 1, stride))
    max_patches = total_y_steps * total_x_steps

    target_count = target_samples if (target_samples and target_samples > 0) else max_patches
    print(f"[Extractor] Target patches: {target_count} ({total_x_steps} cols x {total_y_steps} rows, stride={stride})")

    saved_count = 0
    metadata_records = []
    total_strips = len(range(0, win_h - CROP_PATCH_SIZE + 1, strip_height))
    strip_idx = 0

    with rasterio.Env(**gdal_env_opts):
        with ThreadPoolExecutor(max_workers=min(num_workers, 10)) as executor:
            for y_start in range(0, win_h - CROP_PATCH_SIZE + 1, strip_height):
                if target_samples and target_samples > 0 and saved_count >= target_samples:
                    break

                strip_idx += 1
                curr_h = min(strip_height, win_h - y_start)

                # Check if all patches for this strip are already saved on disk (Auto-Resume)
                y_steps = len(range(0, curr_h - CROP_PATCH_SIZE + 1, stride))
                x_steps = len(range(0, win_w - CROP_PATCH_SIZE + 1, stride))
                strip_expected = y_steps * x_steps

                first_file = os.path.join(output_dir, f"patch_{saved_count:05d}.pt")
                last_file = os.path.join(output_dir, f"patch_{saved_count + strip_expected - 1:05d}.pt")

                if os.path.exists(first_file) and os.path.exists(last_file):
                    print(f"[Strip {strip_idx}/{total_strips}] Already processed. Resuming: skipped {strip_expected} patches ({saved_count + strip_expected}/{target_count})")
                    for y_local in range(0, curr_h - CROP_PATCH_SIZE + 1, stride):
                        global_y = y_start + y_local
                        for x in range(0, win_w - CROP_PATCH_SIZE + 1, stride):
                            patch_id = f"patch_{saved_count:05d}"
                            metadata_records.append({
                                "id": patch_id,
                                "file": f"{patch_id}.pt",
                                "y": int(global_y),
                                "x": int(x)
                            })
                            saved_count += 1
                    continue

                strip_win = Window(win_col_off, win_row_off + y_start, win_w, curr_h)
                strip_bounds = rasterio.windows.bounds(strip_win, ref_transform)

                print(f"[Strip {strip_idx}/{total_strips}] Y={y_start}:{y_start + curr_h} | Streaming {T} dates x 10 bands ({len(S2_10BANDS) * T} bands)...")
                strip_t0 = time.time()
                strip_data = np.zeros((T, 10, curr_h, win_w), dtype=np.float32)

                tasks = [
                    (t_idx, b_idx, scene["assets"][b_name], strip_bounds, curr_h, win_w)
                    for t_idx, scene in enumerate(tile_scenes)
                    for b_idx, b_name in enumerate(S2_10BANDS)
                ]

                failures = 0
                for future in as_completed([executor.submit(fetch_single_band, item) for item in tasks]):
                    t_idx, b_idx, arr, err = future.result()
                    strip_data[t_idx, b_idx] = arr
                    if err:
                        failures += 1

                elapsed = time.time() - strip_t0
                status_extra = f" ({failures} band retries failed)" if failures else ""
                print(f"[Strip {strip_idx}/{total_strips}] Fetched {len(tasks)} bands in {elapsed:.1f}s{status_extra}")

                # Normalize 10 bands: (x - mu) / sigma
                strip_data_norm = normalizer(strip_data)

                # Extract and save 24x24 patches
                strip_saved = 0
                for y_local in range(0, curr_h - CROP_PATCH_SIZE + 1, stride):
                    global_y = y_start + y_local
                    for x in range(0, win_w - CROP_PATCH_SIZE + 1, stride):
                        # Use ascontiguousarray to prevent saving underlying strip buffer (~582KB per patch)
                        patch_img = np.ascontiguousarray(
                            strip_data_norm[:, :, y_local:y_local + CROP_PATCH_SIZE, x:x + CROP_PATCH_SIZE]
                        )
                        patch_lbl = np.ascontiguousarray(
                            label_raster[global_y:global_y + CROP_PATCH_SIZE, x:x + CROP_PATCH_SIZE]
                        )

                        patch_id = f"patch_{saved_count:05d}"
                        file_path = os.path.join(output_dir, f"{patch_id}.pt")

                        torch.save({
                            'image': torch.from_numpy(patch_img).float(),
                            'doy': torch.from_numpy(norm_doys).float().clone(),
                            'label': torch.from_numpy(patch_lbl).long()
                        }, file_path)

                        metadata_records.append({
                            "id": patch_id,
                            "file": f"{patch_id}.pt",
                            "y": int(global_y),
                            "x": int(x)
                        })

                        saved_count += 1
                        strip_saved += 1
                        if target_samples and target_samples > 0 and saved_count >= target_samples:
                            break
                    if target_samples and target_samples > 0 and saved_count >= target_samples:
                        break

                print(f"[Strip {strip_idx}] Saved {strip_saved} patches. Progress: {saved_count}/{target_count}")
                del strip_data
                del strip_data_norm

    meta_file = os.path.join(output_dir, "metadata.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2)

    print(f"\n[Extractor] Completed. Successfully saved {saved_count} patches to: {output_dir}")
    print(f"[Extractor] Metadata catalog written to: {meta_file}")

    return {
        "total_saved": saved_count,
        "timestamps": T,
        "window": (win_h, win_w),
        "output_dir": output_dir,
        "metadata_file": meta_file
    }


def main():
    parser = argparse.ArgumentParser(description="Extract Sentinel-2 24x24 multi-temporal patches")
    parser.add_argument("--metadata", type=str, default="data/raw/planetary_2022/scenes_metadata.json",
                        help="Path to scenes_metadata.json")
    parser.add_argument("--output_dir", type=str, default="data/processed/patches",
                        help="Directory to export .pt patches (default: data/processed/patches)")
    parser.add_argument("--target_samples", type=int, default=0,
                        help="Max patches to export (0 = full AOI extent)")
    parser.add_argument("--max_timestamps", type=int, default=25,
                        help="Number of multi-temporal scenes to use (default: 25)")
    parser.add_argument("--rpg_file", type=str, default=None,
                        help="Path to real RPG 2022 vector file (.shp / .gpkg / .geojson)")
    parser.add_argument("--stride", type=int, default=24,
                        help="Sliding stride in pixels (default: 24)")
    parser.add_argument("--bbox", nargs=4, type=float, default=[1.35, 47.88, 2.27, 48.54],
                        help="AOI BBOX: min_lon min_lat max_lon max_lat (default: 60k patches)")
    parser.add_argument("--strip_height", type=int, default=252,
                        help="Strip chunk height for safe RAM footprint (default: 252)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent worker threads (default: 8)")

    args = parser.parse_args()
    res = process_and_export_patches(
        metadata_json=args.metadata,
        output_dir=args.output_dir,
        target_samples=args.target_samples if args.target_samples > 0 else None,
        max_timestamps=args.max_timestamps,
        rpg_vector_file=args.rpg_file,
        stride=args.stride,
        bbox=tuple(args.bbox),
        strip_height=args.strip_height,
        num_workers=args.workers
    )
    print(f"[Summary] {res}")


if __name__ == "__main__":
    main()
