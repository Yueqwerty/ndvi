from loguru import logger
import numpy as np
from pathlib import Path
from typing import List, Tuple
import rasterio
from rasterio.transform import from_bounds

def load_monthly_ndvi_files(month: int, years: List[int], sub_area_number: int, data_dir: Path) -> List[np.ndarray]:
    """
    Load NDVI data for a given month and sub-area.

    Parameters
    ----------
    month : int
        Target month.
    years : List[int]
        List of target years.
    sub_area_number : int
        Sub-area number.
    data_dir : Path
        Base directory for data.

    Returns
    -------
    List[np.ndarray]
        List of NDVI arrays.
    """
    ndvi_list = []
    for year in years:
        ndvi_path = data_dir / "raw" / "ndvi" / f"{year}-{month:02d}" / f"sub_area_{sub_area_number}" / "ndvi_monthly.npy"
        if ndvi_path.exists():
            try:
                ndvi = np.load(ndvi_path)
                ndvi_list.append(ndvi)
                logger.info(f"Loaded NDVI data from {ndvi_path}")
            except Exception as e:
                logger.error(f"Failed to load NDVI file {ndvi_path}: {e}")
        else:
            logger.warning(f"NDVI file not found: {ndvi_path}")
    return ndvi_list

def compute_statistics(ndvi: np.ndarray) -> dict:
    """
    Compute statistical metrics for an NDVI array.

    Parameters
    ----------
    ndvi : np.ndarray
        NDVI array.

    Returns
    -------
    dict
        Dictionary containing statistical metrics.
    """
    return {
        "mean_ndvi": float(np.nanmean(ndvi)),
        "median_ndvi": float(np.nanmedian(ndvi)),
        "max_ndvi": float(np.nanmax(ndvi)),
        "min_ndvi": float(np.nanmin(ndvi)),
        "std_dev_ndvi": float(np.nanstd(ndvi))
    }

def save_ndvi_as_geotiff(
    ndvi_array: np.ndarray, 
    output_path: Path, 
    sub_area_bounds: Tuple[float, float, float, float], 
    reference_crs: str = "EPSG:4326"
):
    """
    Save NDVI array as a GeoTIFF file.

    Parameters
    ----------
    ndvi_array : np.ndarray
        NDVI array to save.
    output_path : Path
        Path to save the GeoTIFF file.
    sub_area_bounds : Tuple[float, float, float, float]
        Bounds of the area in the format (min_lon, min_lat, max_lon, max_lat).
    reference_crs : str, optional
        Coordinate reference system, by default "EPSG:4326".
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate transformation matrix based on the bounds
        transform = from_bounds(
            west=sub_area_bounds[0],
            south=sub_area_bounds[1],
            east=sub_area_bounds[2],
            north=sub_area_bounds[3],
            width=ndvi_array.shape[1],
            height=ndvi_array.shape[0]
        )

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=ndvi_array.shape[0],
            width=ndvi_array.shape[1],
            count=1,
            dtype=ndvi_array.dtype,
            crs=reference_crs,
            transform=transform,
            nodata=np.nan,
            compress="lzw"
        ) as dst:
            dst.write(ndvi_array, 1)

        print(f"NDVI GeoTIFF saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving NDVI as GeoTIFF: {e}")