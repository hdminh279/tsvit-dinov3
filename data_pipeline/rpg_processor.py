"""
Processor for agricultural parcel polygons (RPG 2022 - Registre Parcellaire Graphique).
Handles clipping to AOI, crop grouping, and rasterization to match Sentinel-2 image grid.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import box, shape, Polygon, MultiPolygon
from shapely.ops import clip_by_rect

from .constants import RPG_2022_GROUP_MAP, NUM_CLASSES, CROP_CLASSES_EN


class RPGProcessor:
    """
    Loads, groups, and rasterizes RPG 2022 agricultural parcel polygons.
    """
    def __init__(self, group_mapping: Optional[Dict[str, int]] = None):
        self.group_mapping = RPG_2022_GROUP_MAP if group_mapping is None else group_mapping

    def map_crop_code(self, raw_code: Union[str, int]) -> int:
        """Map raw RPG crop/group code to target class integer (0 to NUM_CLASSES - 1)."""
        code_str = str(raw_code).strip()
        return self.group_mapping.get(code_str, 0)  # Default to 0 (Background / other)

    def rasterize_parcels(
        self,
        shapes_and_codes: List[Tuple[Union[Polygon, MultiPolygon], Union[str, int]]],
        bounds: Tuple[float, float, float, float],  # (min_lon, min_lat, max_lon, max_lat)
        out_shape: Tuple[int, int],  # (H, W)
        fill: int = 0
    ) -> np.ndarray:
        """
        Rasterizes list of (geometry, crop_code) onto satellite grid.
        
        Args:
            shapes_and_codes: List of tuples (polygon_geometry, crop_code)
            bounds: Bounding coordinates (minx, miny, maxx, maxy)
            out_shape: (H, W) target raster dimensions
            fill: Background label value (default 0)
        Returns:
            2D numpy array [H, W] of crop class integer labels
        """
        H, W = out_shape
        minx, miny, maxx, maxy = bounds
        transform = from_bounds(minx, miny, maxx, maxy, W, H)

        # Prepare geometries with mapped integer values
        shapes_to_rasterize = []
        aoi_box = box(minx, miny, maxx, maxy)

        for geom, code in shapes_and_codes:
            if not geom.is_valid:
                geom = geom.buffer(0)
            if not geom.intersects(aoi_box):
                continue
            clipped_geom = clip_by_rect(geom, minx, miny, maxx, maxy)
            if clipped_geom.is_empty:
                continue

            mapped_class = self.map_crop_code(code)
            shapes_to_rasterize.append((clipped_geom, mapped_class))

        if not shapes_to_rasterize:
            return np.full(out_shape, fill, dtype=np.int64)

        raster = rasterize(
            shapes=shapes_to_rasterize,
            out_shape=out_shape,
            transform=transform,
            fill=fill,
            dtype=np.int64,
            all_touched=False
        )
        return raster

    @staticmethod
    def generate_synthetic_aoi_parcels(
        bounds: Tuple[float, float, float, float],
        grid_rows: int = 6,
        grid_cols: int = 6,
        seed: int = 42
    ) -> List[Tuple[Polygon, str]]:
        """
        Generates realistic synthetic agricultural parcels within an AOI for testing and verification.
        """
        rng = np.random.default_rng(seed)
        minx, miny, maxx, maxy = bounds
        dx = (maxx - minx) / grid_cols
        dy = (maxy - miny) / grid_rows

        valid_codes = list(RPG_2022_GROUP_MAP.keys())
        parcels = []

        for r in range(grid_rows):
            for c in range(grid_cols):
                # Add slight jitter to parcel boundaries to mimic organic field shapes
                jitter = 0.05
                p_minx = minx + c * dx + rng.uniform(-jitter * dx, jitter * dx)
                p_miny = miny + r * dy + rng.uniform(-jitter * dy, jitter * dy)
                p_maxx = minx + (c + 1) * dx + rng.uniform(-jitter * dx, jitter * dx)
                p_maxy = miny + (r + 1) * dy + rng.uniform(-jitter * dy, jitter * dy)

                poly = box(p_minx, p_miny, p_maxx, p_maxy)
                code = rng.choice(valid_codes)
                parcels.append((poly, code))

        return parcels

    @staticmethod
    def load_parcels_from_file(
        file_path: str,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        code_field: str = "CODE_GROUP"
    ) -> List[Tuple[Union[Polygon, MultiPolygon], str]]:
        """
        Loads actual agricultural parcels from an RPG 2022 file (.geojson, .shp, or .gpkg).
        
        Args:
            file_path: Path to the vector file (.geojson, .shp, or .gpkg)
            bounds: Optional AOI bounding box (min_lon, min_lat, max_lon, max_lat) to pre-filter
            code_field: Attribute field containing crop code (default "CODE_GROUP" or fallback to "CODE_CULTU")
        Returns:
            List of (polygon_geometry, crop_code) tuples
        """
        import os
        import json
        parcels = []
        aoi_box = box(*bounds) if bounds is not None else None

        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".geojson", ".json"]:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data.get("features", [])
            for feat in features:
                props = feat.get("properties", {})
                code = str(props.get(code_field, props.get("CODE_CULTU", "0")))
                geom_dict = feat.get("geometry")
                if not geom_dict:
                    continue
                geom = shape(geom_dict)
                if not geom.is_valid:
                    geom = geom.buffer(0)
                if aoi_box is not None and not geom.intersects(aoi_box):
                    continue
                parcels.append((geom, code))
        else:
            # Fallback to fiona or geopandas if installed for ESRI Shapefile / GeoPackage
            try:
                import geopandas as gpd
                gdf = gpd.read_file(file_path)
                if bounds is not None:
                    gdf = gdf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
                target_field = code_field if code_field in gdf.columns else ("CODE_CULTU" if "CODE_CULTU" in gdf.columns else gdf.columns[0])
                for _, row in gdf.iterrows():
                    geom = row.geometry
                    code = str(row[target_field])
                    if not geom.is_valid:
                        geom = geom.buffer(0)
                    parcels.append((geom, code))
            except ImportError:
                raise ImportError(
                    f"To load {ext} files, please install geopandas: `pip install geopandas` or convert to GeoJSON."
                )

        return parcels

