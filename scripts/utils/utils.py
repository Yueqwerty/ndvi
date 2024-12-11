from pathlib import Path
from typing import Tuple
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from loguru import logger

def save_ndvi_as_geotiff(ndvi: np.ndarray, output_path: Path, sub_area_bounds: Tuple[float, float, float, float]) -> bool:
    """
    Saves the processed NDVI data to a GeoTIFF file.

    Parameters
    ----------
    ndvi : np.ndarray
        NDVI data as a NumPy array.
    output_path : Path
        Path to save the GeoTIFF file.
    sub_area_bounds : Tuple[float, float, float, float]
        (min_lon, min_lat, max_lon, max_lat) in WGS84.

    Returns
    -------
    bool
        True if saved successfully, False otherwise.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width = ndvi.shape[1]
        height = ndvi.shape[0]

        transform = from_bounds(
            west=sub_area_bounds[0],
            south=sub_area_bounds[1],
            east=sub_area_bounds[2],
            north=sub_area_bounds[3],
            width=width,
            height=height
        )

        # Using EPSG:4326 for WGS84 lat/lon
        crs = "EPSG:4326"

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=ndvi.dtype,
            crs=crs,
            transform=transform,
            compress='lzw',
            nodata=np.nan
        ) as dst:
            dst.write(ndvi, 1)

        logger.info(f"Processed NDVI saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving processed NDVI to {output_path}: {e}")
        return False
