import json
from pathlib import Path
from typing import Dict, List, Tuple
from shapely.geometry import Polygon
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from loguru import logger


def load_evalscript(evalscript_path: Path) -> str:
    """
    Load Evalscript from a file.

    Parameters
    ----------
    evalscript_path : Path
        Path to the Evalscript file.

    Returns
    -------
    str
        Content of the Evalscript.
    """
    try:
        with open(evalscript_path, 'r') as file:
            evalscript = file.read()
        logger.info(f"Loaded Evalscript from {evalscript_path}")
        return evalscript
    except Exception as e:
        logger.error(f"Failed to load Evalscript: {e}")
        sys.exit(1)


def save_ndvi_as_geotiff(ndvi: np.ndarray, geotiff_path: Path, bounds: List[float], crs: str = "EPSG:4326") -> bool:
    """
    Save NDVI array as a GeoTIFF file.

    Parameters
    ----------
    ndvi : np.ndarray
        NDVI array.
    geotiff_path : Path
        Path to save the GeoTIFF.
    bounds : List[float]
        Bounds of the image in the format [min_lon, min_lat, max_lon, max_lat].
    crs : str, optional
        Coordinate Reference System, by default "EPSG:4326".

    Returns
    -------
    bool
        True if saved successfully, False otherwise.
    """
    try:
        min_lon, min_lat, max_lon, max_lat = bounds
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, ndvi.shape[1], ndvi.shape[0])
        new_dataset = rasterio.open(
            geotiff_path,
            'w',
            driver='GTiff',
            height=ndvi.shape[0],
            width=ndvi.shape[1],
            count=1,
            dtype=ndvi.dtype,
            crs=crs,
            transform=transform,
            nodata=np.nan
        )
        new_dataset.write(ndvi, 1)
        new_dataset.close()
        logger.info(f"GeoTIFF saved successfully at {geotiff_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving processed NDVI to {geotiff_path}: {e}")
        return False


def divide_aoi_grid(aoi_polygon: Polygon, tile_size: float) -> List[Polygon]:
    """
    Divide the Area of Interest (AOI) into a grid of specified tile size.

    Parameters
    ----------
    aoi_polygon : shapely.geometry.Polygon
        The AOI polygon.
    tile_size : float
        Size of each tile in the same units as the AOI's CRS (e.g., meters).

    Returns
    -------
    List[shapely.geometry.Polygon]
        List of polygon tiles covering the AOI.
    """
    from shapely.geometry import box
    minx, miny, maxx, maxy = aoi_polygon.bounds
    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tile = box(x, y, x + tile_size, y + tile_size)
            intersection = aoi_polygon.intersection(tile)
            if not intersection.is_empty:
                tiles.append(intersection)
            y += tile_size
        x += tile_size
    return tiles


def calculate_output_dimensions(polygon: Polygon, resolution: float = 10.0) -> Tuple[int, int]:
    """
    Calculate the output image dimensions based on the polygon's area and desired resolution.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The polygon defining the sub-area.
    resolution : float, optional
        Spatial resolution in meters per pixel, by default 10.0.

    Returns
    -------
    Tuple[int, int]
        Width and height in pixels.
    """
    minx, miny, maxx, maxy = polygon.bounds
    width_m = maxx - minx
    height_m = maxy - miny
    width_px = int(width_m / resolution)
    height_px = int(height_m / resolution)
    return width_px, height_px


def load_seasons_config(config_path: Path) -> Dict[str, List[int]]:
    """
    Load seasons configuration from a JSON file.

    Parameters
    ----------
    config_path : Path
        Path to the seasons configuration JSON file.

    Returns
    -------
    Dict[str, List[int]]
        Dictionary mapping season names to lists of months.
    """
    try:
        with open(config_path, 'r') as file:
            seasons = json.load(file)
        logger.info(f"Loaded seasons configuration from {config_path}")
        return seasons
    except Exception as e:
        logger.error(f"Failed to load seasons configuration: {e}")
        sys.exit(1)


def compute_statistics(ndvi: np.ndarray) -> Dict[str, float]:
    """
    Compute statistical metrics for an NDVI array.

    Parameters
    ----------
    ndvi : np.ndarray
        NDVI array.

    Returns
    -------
    Dict[str, float]
        Dictionary containing statistical metrics.
    """
    stats = {
        "mean_ndvi": float(np.nanmean(ndvi)),
        "median_ndvi": float(np.nanmedian(ndvi)),
        "max_ndvi": float(np.nanmax(ndvi)),
        "min_ndvi": float(np.nanmin(ndvi)),
        "std_dev_ndvi": float(np.nanstd(ndvi))
    }
    return stats


def save_statistics(statistics: Dict[str, float], stats_path: Path) -> None:
    """
    Save statistical metrics as a JSON file.

    Parameters
    ----------
    statistics : Dict[str, float]
        Statistical metrics.
    stats_path : Path
        Path to save the JSON file.
    """
    try:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=4)
        logger.info(f"Statistics saved to {stats_path}")
    except IOError as e:
        logger.error(f"Error saving statistics to {stats_path}: {e}")


def aggregate_seasonal_ndvi(best_images: List[np.ndarray], method: str = "mean") -> np.ndarray:
    """
    Aggregate selected NDVI images to form a seasonal representation.

    Parameters
    ----------
    best_images : List[np.ndarray]
        List of top NDVI arrays with the highest number of valid pixels.
    method : str, optional
        Aggregation method ('mean', 'median', 'max'), by default "mean".

    Returns
    -------
    np.ndarray
        Aggregated seasonal NDVI array.
    """
    if not best_images:
        logger.error("No images to aggregate.")
        return np.array([])

    # Stack images along a new axis
    stacked_ndvi = np.stack(best_images, axis=0)

    # Aggregate using the specified method
    if method == "mean":
        seasonal_ndvi = np.nanmean(stacked_ndvi, axis=0)
    elif method == "median":
        seasonal_ndvi = np.nanmedian(stacked_ndvi, axis=0)
    elif method == "max":
        seasonal_ndvi = np.nanmax(stacked_ndvi, axis=0)
    else:
        logger.error(f"Invalid aggregation method: {method}. Using 'mean' by default.")
        seasonal_ndvi = np.nanmean(stacked_ndvi, axis=0)

    logger.info(f"Seasonal NDVI aggregation completed using method '{method}'.")
    return seasonal_ndvi


def select_best_images(ndvi_list: List[np.ndarray], top_n: int = 3) -> List[np.ndarray]:
    """
    Select the best NDVI images based on the number of valid pixels.

    Parameters
    ----------
    ndvi_list : List[np.ndarray]
        List of NDVI arrays for the season.
    top_n : int, optional
        Number of top images to select, by default 3.

    Returns
    -------
    List[np.ndarray]
        List of top NDVI arrays with the highest number of valid pixels.
    """
    if not ndvi_list:
        logger.error("NDVI list is empty. Cannot select images.")
        return []

    quality_scores = [np.count_nonzero(~np.isnan(ndvi)) for ndvi in ndvi_list]
    sorted_indices = np.argsort(quality_scores)[::-1]  # Descending order
    best_indices = sorted_indices[:top_n]
    best_images = [ndvi_list[i] for i in best_indices]
    logger.info(f"Selected top {len(best_images)} images with the highest number of valid pixels.")
    return best_images

def load_seasonal_ndvi(season_months: List[str], sub_area: int, data_dir: Path) -> List[np.ndarray]:
    """
    Load aggregated seasonal NDVI files for a specific sub-area.

    Parameters
    ----------
    season_months : List[str]
        List of month strings in 'YYYY-MM' format included in the season.
    sub_area : int
        Sub-area number.
    data_dir : Path
        Path to the data directory.

    Returns
    -------
    List[np.ndarray]
        List of NDVI arrays.
    """
    ndvi_list = []
    for month in season_months:
        ndvi_file = data_dir / "raw" / "ndvi" / month / f"sub_area_{sub_area}" / "ndvi_monthly.npy"
        if ndvi_file.exists():
            ndvi = np.load(ndvi_file)
            ndvi_list.append(ndvi)
            logger.debug(f"Loaded NDVI from {ndvi_file}")
        else:
            logger.warning(f"NDVI file not found: {ndvi_file}")
    return ndvi_list