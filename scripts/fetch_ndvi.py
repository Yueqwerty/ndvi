#!/usr/bin/env python3
"""
Enhanced Fetch NDVI Script with Cloud Masking and Seasonal Best Image Selection

This script fetches Landsat-8 NDVI data for specified seasons and sub-areas,
applies cloud masking using a custom Evalscript, evaluates image quality based on
validated (cloud-free) pixels, and selects the best images per season for further analysis.

Usage:
    python scripts/fetch_ndvi.py <YEAR> <SEASON> [--sub_areas SUB_AREA_NUMBERS] [--evalscript EVALSCRIPT_PATH]
"""

import argparse
import asyncio
import calendar
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import aiohttp
import numpy as np
import rasterio
from loguru import logger
from shapely.geometry import Polygon, box
from pyproj import Transformer
from dotenv import load_dotenv

from utils.utils import save_ndvi_as_geotiff, load_evalscript, divide_aoi_grid, calculate_output_dimensions

# Load environment variables from .env file
load_dotenv()

# Define project directories
SCRIPT_DIR = Path(__file__).resolve().parent  # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent  # proyecto_ndvi/
DATA_DIR = PROJECT_ROOT / 'data'  # proyecto_ndvi/data/
LOGS_DIR = PROJECT_ROOT / 'logs'  # proyecto_ndvi/logs/
EVALSCRIPT_DIR = SCRIPT_DIR / 'evalscripts'  # scripts/evalscripts/

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
EVALSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# Configure Logging
LOG_FILE = LOGS_DIR / 'fetch_ndvi.log'
logger.remove()  # Remove default logger to prevent duplication
logger.add(LOG_FILE, rotation="1 MB", retention="7 days", level="DEBUG")
logger.add(sys.stdout, level="INFO")  # Add console logging

# Sentinel Hub Configuration
CLIENT_ID = os.getenv("SENTINELHUB_CLIENT_ID")
CLIENT_SECRET = os.getenv("SENTINELHUB_CLIENT_SECRET")
TOKEN_URL = os.getenv("SENTINELHUB_TOKEN_URL", "https://services.sentinel-hub.com/oauth/token")
PROCESS_URL = os.getenv("SENTINELHUB_PROCESS_URL", "https://services.sentinel-hub.com/api/v1/process")

# Verify Credentials
if not CLIENT_ID or not CLIENT_SECRET:
    logger.error("Environment variables SENTINELHUB_CLIENT_ID and SENTINELHUB_CLIENT_SECRET must be set.")
    sys.exit(1)

# CRS Transformers
transformer_to_proj = Transformer.from_crs("epsg:4326", "epsg:32719", always_xy=True)
transformer_to_wgs84 = Transformer.from_crs("epsg:32719", "epsg:4326", always_xy=True)

# Configuration Parameters
MAX_CONCURRENT_REQUESTS = 2  # Adjust based on API limits
MAX_CONSECUTIVE_FAILURES = 3  # Limit to skip problematic areas quickly
BACKOFF_FACTOR = 2  # Exponential backoff factor

# Define the seasons for the Southern Hemisphere
SEASONS = {
    "spring": [9, 10, 11],   # September, October, November
    "summer": [12, 1, 2],    # December, January, February
    "autumn": [3, 4, 5],     # March, April, May
    "winter": [6, 7, 8],     # June, July, August
}

async def obtain_token(session: aiohttp.ClientSession) -> Optional[str]:
    """
    Obtain Sentinel Hub access token.
    """
    token_data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        async with session.post(TOKEN_URL, data=token_data, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                token = data.get("access_token")
                logger.info("Successfully obtained access token.")
                return token
            else:
                text = await response.text()
                logger.error(f"Failed to obtain token: {response.status} {text}")
                return None
    except Exception as e:
        logger.error(f"Exception while obtaining token: {e}")
        return None

async def request_ndvi_landsat8(
    session: aiohttp.ClientSession,
    token: str,
    polygon_coords: List[Tuple[float, float]],
    date: str,
    evalscript: str,
    sub_area_number: int,
    width_pixels: int,
    height_pixels: int,
    consecutive_failures: int,
    attempt: int = 1
) -> Tuple[Optional[bytes], int]:
    """
    Request Landsat-8 NDVI data from Sentinel Hub API for a specific sub-area and date.
    Implements exponential backoff for retries.
    """
    request_payload = {
        "input": {
            "bounds": {
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"  # Use WGS84 CRS
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon_coords]
                }
            },
            "data": [
                {
                    "type": "landsat-ot-l1",  # Sentinel Hub data type
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date}T00:00:00Z",
                            "to": f"{date}T23:59:59Z"
                        }
                    }
                }
            ]
        },
        "evalscript": evalscript,
        "output": {
            "width": width_pixels,
            "height": height_pixels,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff",
                        "samples": 1,
                        "dataType": "FLOAT32"
                    }
                }
            ]
        }
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "*/*"
    }

    try:
        logger.debug(f"Requesting NDVI for Sub-area {sub_area_number} on {date}, Attempt {attempt}...")
        async with session.post(PROCESS_URL, json=request_payload, headers=headers, timeout=600) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if response.status == 200 and ("image/tiff" in content_type or "application/octet-stream" in content_type):
                content = await response.read()
                logger.debug(f"Successfully fetched NDVI for Sub-area {sub_area_number} on {date}.")
                return content, 0  # Reset failure count on success
            elif response.status == 429:
                # Handle rate limiting
                logger.warning(f"Rate limit exceeded for Sub-area {sub_area_number} on {date}.")
                return None, consecutive_failures + 1
            elif response.status >= 500:
                # Server error, may retry
                logger.warning(f"Server error {response.status} for Sub-area {sub_area_number} on {date}.")
                return None, consecutive_failures + 1
            else:
                text = await response.text()
                logger.error(f"Failed request for NDVI for Sub-area {sub_area_number} on {date}: {response.status} {text}")
                # Optionally, save error response for debugging
                error_path = DATA_DIR / "raw" / "ndvi" / date / f"sub_area_{sub_area_number}" / "error_response.txt"
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(text)
                return None, consecutive_failures + 1
    except asyncio.TimeoutError:
        logger.error(f"NDVI request timed out for Sub-area {sub_area_number} on {date}.")
        if attempt < MAX_CONSECUTIVE_FAILURES:
            sleep_time = BACKOFF_FACTOR ** attempt
            logger.info(f"Retrying after {sleep_time} seconds...")
            await asyncio.sleep(sleep_time)
            return await request_ndvi_landsat8(
                session,
                token,
                polygon_coords,
                date,
                evalscript,
                sub_area_number,
                width_pixels,
                height_pixels,
                consecutive_failures,
                attempt + 1
            )
        else:
            logger.error(f"Exceeded maximum retries for Sub-area {sub_area_number} on {date}.")
            return None, consecutive_failures + 1
    except aiohttp.ClientError as client_err:
        logger.error(f"Client error during NDVI request for Sub-area {sub_area_number} on {date}: {client_err}.")
        return None, consecutive_failures + 1
    except Exception as e:
        logger.error(f"Unexpected error during NDVI request for Sub-area {sub_area_number} on {date}: {e}")
        return None, consecutive_failures + 1

def refine_cloud_mask(cloud_mask: np.ndarray) -> np.ndarray:
    """
    Refine the cloud mask using morphological operations.

    Parameters
    ----------
    cloud_mask : np.ndarray
        Binary cloud mask array.

    Returns
    -------
    np.ndarray
        Refined cloud mask array.
    """
    from scipy.ndimage import binary_erosion, binary_dilation, median_filter

    # Remove small artifacts
    cloud_mask = binary_erosion(cloud_mask, structure=np.ones((3,3))).astype(int)
    cloud_mask = binary_dilation(cloud_mask, structure=np.ones((3,3))).astype(int)
    
    # Apply median filter to smooth edges
    cloud_mask = median_filter(cloud_mask, size=3)
    
    return cloud_mask

def evaluate_image_quality(ndvi: np.ndarray) -> int:
    """
    Evaluate the quality of an NDVI image based on the number of valid pixels.

    Parameters
    ----------
    ndvi : np.ndarray
        NDVI array.

    Returns
    -------
    int
        Number of valid (cloud-free) pixels.
    """
    return np.count_nonzero(~np.isnan(ndvi))

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

    quality_scores = [evaluate_image_quality(ndvi) for ndvi in ndvi_list]
    sorted_indices = np.argsort(quality_scores)[::-1]  # Descending order
    best_indices = sorted_indices[:top_n]
    best_images = [ndvi_list[i] for i in best_indices]
    logger.info(f"Selected top {len(best_images)} images with the highest number of valid pixels.")
    return best_images

def aggregate_seasonal_ndvi(best_images: List[np.ndarray], method: str = "mean") -> np.ndarray:
    """
    Aggregate selected NDVI images to form a seasonal representation.

    Parameters
    ----------
    best_images : List[np.ndarray]
        List of top NDVI arrays.
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

async def process_sub_area(
    session: aiohttp.ClientSession,
    sub_area: Dict,
    dates: List[str],
    evalscript: str,
    token: str,
    raw_ndvi_path: Path
) -> None:
    """
    Process a single sub-area by fetching, masking clouds, and aggregating NDVI data.

    Parameters
    ----------
    session : aiohttp.ClientSession
        The aiohttp session to use for requests.
    sub_area : Dict
        Dictionary containing sub-area details.
    dates : List[str]
        List of date strings in 'YYYY-MM-DD' format.
    evalscript : str
        The Evalscript for NDVI calculation with cloud masking.
    token : str
        Sentinel Hub access token.
    raw_ndvi_path : Path
        Path to save the aggregated monthly NDVI data.
    """
    sub_area_number = sub_area['number']
    polygon_coords = sub_area['coords']
    width_pixels = sub_area['width_pixels']
    height_pixels = sub_area['height_pixels']

    ndvi_values = []
    cloud_masks = []
    consecutive_failures = 0  # Initialize failure counter

    for date in dates:
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.warning(f"Maximum consecutive failures reached for Sub-area {sub_area_number}. Skipping remaining dates.")
            break

        ndvi_landsat, consecutive_failures = await request_ndvi_landsat8(
            session,
            token,
            polygon_coords,
            date,
            evalscript,
            sub_area_number,
            width_pixels,
            height_pixels,
            consecutive_failures
        )

        if ndvi_landsat:
            try:
                with rasterio.MemoryFile(ndvi_landsat) as memfile:
                    with memfile.open() as dataset:
                        ndvi_data = dataset.read(1)
                        # Assuming cloud pixels are marked as 1 and no cloud as 0
                        cloud_mask = np.where(ndvi_data == 1, 1, 0)
                        # Refine the cloud mask
                        cloud_mask = refine_cloud_mask(cloud_mask)
                        # Mask out cloud pixels
                        ndvi_valid = np.where(cloud_mask == 1, np.nan, ndvi_data)
                        ndvi_values.append(ndvi_valid)
                        cloud_masks.append(cloud_mask)
                logger.debug(f"NDVI and cloud mask data for Sub-area {sub_area_number} on {date} aggregated.")
            except Exception as e:
                logger.error(f"Error processing NDVI data for Sub-area {sub_area_number} on {date}: {e}")
        else:
            logger.warning(f"No NDVI data obtained for Sub-area {sub_area_number} on {date}.")

    if not ndvi_values:
        logger.error(f"No valid NDVI data found for Sub-area {sub_area_number} in the specified dates.")
        return

    # Aggregate NDVI data using masked values (e.g., mean)
    aggregated_ndvi = np.nanmean(ndvi_values, axis=0)
    logger.info(f"Aggregated NDVI data for Sub-area {sub_area_number}.")

    # Validate NDVI data
    if not np.isfinite(aggregated_ndvi).any():
        logger.error(f"Aggregated NDVI data invalid for Sub-area {sub_area_number}. Skipping save.")
        return

    # Save aggregated NDVI as `.npy`
    ndvi_npy_path = raw_ndvi_path / f"sub_area_{sub_area_number}" / "ndvi_monthly.npy"
    try:
        ndvi_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(ndvi_npy_path, aggregated_ndvi)
        logger.info(f"Aggregated monthly NDVI saved to {ndvi_npy_path}")
    except IOError as io_err:
        logger.error(f"Error saving aggregated NDVI for Sub-area {sub_area_number}: {io_err}")

    # Optionally: Save aggregated NDVI as GeoTIFF
    sub_area_bounds = sub_area.get('bounds')
    if sub_area_bounds:
        geotiff_path = raw_ndvi_path / f"sub_area_{sub_area_number}" / "ndvi_monthly.tif"
        try:
            save_ndvi_as_geotiff(aggregated_ndvi, geotiff_path, sub_area_bounds)
            logger.info(f"Aggregated monthly NDVI GeoTIFF saved to {geotiff_path}")
        except Exception as e:
            logger.error(f"Error saving aggregated monthly NDVI GeoTIFF for Sub-area {sub_area_number}: {e}")
    else:
        logger.warning(f"Bounds not found for Sub-area {sub_area_number}. Skipping GeoTIFF save.")

async def process_sub_area_with_semaphore(semaphore, session, sub_area, dates, evalscript, token, raw_ndvi_path):
    """
    Wrapper to process a sub-area with semaphore control.
    """
    async with semaphore:
        await process_sub_area(session, sub_area, dates, evalscript, token, raw_ndvi_path)

async def main_async(
    year: int,
    season: str,
    sub_areas: Optional[List[int]],
    top_n: int,
    method: str,
    evalscript_path: Path,
    data_dir: Path,
    output_dir: Path
) -> None:
    """
    Asynchronous main function to process all sub-areas for a given season and perform seasonal aggregation.
    """
    # Load Evalscript
    evalscript = load_evalscript(evalscript_path)

    # Define AOI coordinates (WGS84)
    POLYGON_COORDINATES = [
        [
            (-72.476854, -45.805818),
            (-72.476854, -45.282606),
            (-71.581419, -45.282606),
            (-71.581419, -45.805818),
            (-72.476854, -45.805818)
        ]
    ]

    # Transform AOI to projected CRS
    aoi_coords_proj = [transformer_to_proj.transform(lon, lat) for lon, lat in POLYGON_COORDINATES[0]]
    aoi_polygon_proj = Polygon(aoi_coords_proj)

    if not aoi_polygon_proj.is_valid:
        logger.error("Projected AOI geometry is invalid.")
        sys.exit(1)
    else:
        logger.info("Projected AOI geometry is valid.")

    # Divide AOI into sub-areas using grid
    tile_size = 20000.0  # Tile size in meters
    sub_polygons = divide_aoi_grid(aoi_polygon_proj, tile_size=tile_size)
    logger.info(f"Number of sub-areas: {len(sub_polygons)}")

    # Prepare sub-areas with identifiers and output dimensions
    sub_areas_list = []
    for idx, poly in enumerate(sub_polygons):
        width_pixels, height_pixels = calculate_output_dimensions(poly)
        coords_wgs84 = [transformer_to_wgs84.transform(x, y) for x, y in poly.exterior.coords]
        sub_areas_list.append({
            "number": idx + 1,
            "coords": coords_wgs84,
            "polygon": poly,
            "width_pixels": width_pixels,
            "height_pixels": height_pixels,
            "bounds": poly.bounds  # (minx, miny, maxx, maxy) in projected CRS
        })

    # Filter sub-areas if specified
    if sub_areas:
        sub_areas_list = [sa for sa in sub_areas_list if sa["number"] in sub_areas]
        logger.info(f"Processing sub-areas: {sub_areas}")

    # Create a dictionary to store the bounds of each sub-area
    sub_area_bounds = {}
    for sub_area_dict in sub_areas_list:
        sub_area_number = sub_area_dict['number']
        bounds = sub_area_dict['bounds']
        # Convert projected bounds to WGS84
        min_lon, min_lat = transformer_to_wgs84.transform(bounds[0], bounds[1])
        max_lon, max_lat = transformer_to_wgs84.transform(bounds[2], bounds[3])
        sub_area_bounds[str(sub_area_number)] = [min_lon, min_lat, max_lon, max_lat]

    # Save bounds to a JSON file
    bounds_file = data_dir / 'sub_area_bounds.json'
    bounds_file.parent.mkdir(parents=True, exist_ok=True)
    with open(bounds_file, 'w') as f:
        json.dump(sub_area_bounds, f)
    logger.info(f"Sub-area bounds saved to {bounds_file}")

    # Get all months in the specified season
    months = SEASONS.get(season.lower())
    if not months:
        logger.error(f"Invalid season: {season}")
        sys.exit(1)
    logger.info(f"Months included in {season.capitalize()} {year}: {months}")

    # Handle seasons that span two years (e.g., winter)
    if season.lower() == "winter":
        months_with_years = []
        for m in months:
            if m == 12:
                months_with_years.append((year - 1, m))
            else:
                months_with_years.append((year, m))
    else:
        months_with_years = [(year, m) for m in months]

    # Compile month strings in 'YYYY-MM' format
    season_months = [f"{y}-{m:02d}" for y, m in months_with_years]
    logger.info(f"Season '{season}' months: {season_months}")

    # Define paths to save seasonal data
    for month_str in season_months:
        month_path = data_dir / "raw" / "ndvi" / month_str
        month_path.mkdir(parents=True, exist_ok=True)

    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession() as session:
        token = await obtain_token(session)
        if not token:
            logger.error("Failed to obtain Sentinel Hub access token.")
            sys.exit(1)

        # Process each sub-area concurrently for all relevant months
        tasks = []
        for sub_area_dict in sub_areas_list:
            for y, m in months_with_years:
                month_str = f"{y}-{m:02d}"
                # Get all dates in the month
                num_days = calendar.monthrange(y, m)[1]
                dates = [f"{y}-{m:02d}-{day:02d}" for day in range(1, num_days + 1)]
                raw_ndvi_path = data_dir / "raw" / "ndvi" / month_str
                tasks.append(
                    asyncio.create_task(
                        process_sub_area_with_semaphore(
                            semaphore,
                            session,
                            sub_area_dict,
                            dates,
                            evalscript,
                            token,
                            raw_ndvi_path
                        )
                    )
                )
        await asyncio.gather(*tasks)

    logger.info("NDVI data fetching and aggregation process completed successfully.")

    # Perform seasonal aggregation by selecting the best images
    logger.info(f"Performing seasonal aggregation for {season.capitalize()} {year}.")

    for sub_area_dict in sub_areas_list:
        sub_area_number = sub_area_dict["number"]
        logger.info(f"Aggregating seasonal NDVI for Sub-area {sub_area_number}.")

        # Load NDVI data for all months in the season
        seasonal_ndvi_list = []
        for month_str in season_months:
            ndvi_file = data_dir / "raw" / "ndvi" / month_str / f"sub_area_{sub_area_number}" / "ndvi_monthly.npy"
            if ndvi_file.exists():
                ndvi = np.load(ndvi_file)
                seasonal_ndvi_list.append(ndvi)
                logger.debug(f"NDVI loaded from {ndvi_file}")
            else:
                logger.warning(f"NDVI file not found: {ndvi_file}")

        # Select the best images based on valid pixels
        best_images = select_best_images(seasonal_ndvi_list, top_n=top_n)

        # Aggregate the best images
        aggregated_seasonal_ndvi = aggregate_seasonal_ndvi(best_images, method=method)

        if aggregated_seasonal_ndvi.size == 0:
            logger.error(f"No data aggregated for Sub-area {sub_area_number}. Skipping save.")
            continue

        # Define output path for aggregated seasonal NDVI
        aggregated_output_path = output_dir / "statistics" / f"ndvi_{season}_{year}_sub_area_{sub_area_number}.npy"
        aggregated_output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.save(aggregated_output_path, aggregated_seasonal_ndvi)
            logger.info(f"Aggregated seasonal NDVI saved to {aggregated_output_path}")
        except IOError as io_err:
            logger.error(f"Error saving aggregated seasonal NDVI for Sub-area {sub_area_number}: {io_err}")

        # Optionally, save aggregated NDVI as GeoTIFF
        bounds = sub_area_dict['bounds']
        if bounds:
            geotiff_path = output_dir / "statistics" / f"ndvi_{season}_{year}_sub_area_{sub_area_number}.tif"
            try:
                save_ndvi_as_geotiff(aggregated_seasonal_ndvi, geotiff_path, bounds)
                logger.info(f"Aggregated seasonal NDVI GeoTIFF saved to {geotiff_path}")
            except Exception as e:
                logger.error(f"Error saving aggregated seasonal NDVI GeoTIFF for Sub-area {sub_area_number}: {e}")
        else:
            logger.warning(f"Bounds not found for Sub-area {sub_area_number}. Skipping GeoTIFF save.")

def main():
    """
    Entry point of the fetch_ndvi.py script.
    """
    parser = argparse.ArgumentParser(
        description="Fetch and process Landsat-8 NDVI data with cloud masking and seasonal best image selection."
    )
    parser.add_argument(
        "year",
        type=int,
        help="Target year as an integer (e.g., 2020)."
    )
    parser.add_argument(
        "season",
        type=str,
        choices=SEASONS.keys(),
        help="Season to process (e.g., spring)."
    )
    parser.add_argument(
        "--sub_areas",
        type=int,
        nargs='+',
        help="Sub-area number(s) to process. If not specified, all sub-areas will be processed."
    )
    parser.add_argument(
        "--evalscript",
        type=str,
        default="evalscripts/cloud_masking_evalscript.js",
        help="Path to the Evalscript file for cloud masking."
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=3,
        help="Number of top images to select per season based on validated pixels."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["mean", "median", "max"],
        default="mean",
        help="Aggregation method to use for seasonal NDVI."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to the data directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/statistics",
        help="Path to save the statistics JSON files and aggregated NDVI."
    )

    args = parser.parse_args()

    # Validate year
    if args.year <= 0:
        logger.error("Invalid year provided.")
        sys.exit(1)

    # Validate Evalscript path
    evalscript_path = Path(args.evalscript)
    if not evalscript_path.exists():
        logger.error(f"Evalscript file not found: {evalscript_path}")
        sys.exit(1)

    # Run the asynchronous main function
    asyncio.run(
        main_async(
            year=args.year,
            season=args.season,
            sub_areas=args.sub_areas,
            top_n=args.top_n,
            method=args.method,
            evalscript_path=evalscript_path,
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir)
        )
    )

if __name__ == "__main__":
    main()
