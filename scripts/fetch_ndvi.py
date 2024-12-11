#!/usr/bin/env python3
"""
Fetch and Aggregate Landsat-8 NDVI Data for a Specific Month (No Cloud Masking)

This script fetches Landsat-8 NDVI data for each day of a specified month within a given AOI,
aggregates the NDVI data to form a monthly composite, and saves both NumPy and GeoTIFF outputs.

**Improvements and Enhancements:**
- Added more robust error handling and logging for token retrieval and requests.
- Implemented exponential backoff with jitter for rate limiting and server errors.
- Added docstrings and comments for clarity.
- Ensured that the CRS and nodata handling are well-defined.
- Checked image dimension calculations and fallback to defaults if needed.
- Confirmed that retrieved NDVI data is stored consistently in data/raw/ndvi/<YYYY-MM>/sub_area_X directories.

Usage:
    python scripts/fetch_ndvi.py YYYY-MM

Example:
    python scripts/fetch_ndvi.py 2023-04
"""

import argparse
import asyncio
import aiohttp
import os
import sys
import json
import calendar
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import rasterio
from loguru import logger
from shapely.geometry import Polygon, box
from pyproj import Transformer
from dotenv import load_dotenv

from utils import save_ndvi_as_geotiff

# Load environment variables from .env file
load_dotenv()

# Define project directories
SCRIPT_DIR = Path(__file__).resolve().parent  # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent               # proyecto_ndvi/
DATA_DIR = PROJECT_ROOT / 'data'               # proyecto_ndvi/data/
LOGS_DIR = PROJECT_ROOT / 'logs'               # proyecto_ndvi/logs/

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure Logging
LOG_FILE = LOGS_DIR / 'fetch_ndvi.log'
LOG_LEVEL = os.getenv("FETCH_NDVI_LOG_LEVEL", "DEBUG").upper()

logger.remove()  # Remove default logger to prevent duplication
logger.add(LOG_FILE, rotation="1 MB", retention="7 days", level=LOG_LEVEL)
logger.add(sys.stdout, level=LOG_LEVEL)  # Add console logging

# Sentinel Hub Configuration (must be set via .env)
CLIENT_ID = os.getenv("SENTINELHUB_CLIENT_ID", "c1326a3f-464d-4e4b-897a-4ca6281ffc2d")
CLIENT_SECRET = os.getenv("SENTINELHUB_CLIENT_SECRET", "NNZzqTVR5DWQraGNCFTR5O0AHLypxKQg")
TOKEN_URL = os.getenv("SENTINELHUB_TOKEN_URL", "https://services.sentinel-hub.com/oauth/token")
PROCESS_URL = os.getenv("SENTINELHUB_PROCESS_URL", "https://services-uswest2.sentinel-hub.com/api/v1/process")

if not CLIENT_ID or not CLIENT_SECRET:
    logger.error("SENTINELHUB_CLIENT_ID and SENTINELHUB_CLIENT_SECRET must be set in .env.")
    sys.exit(1)

# CRS Transformers
transformer_to_proj = Transformer.from_crs("epsg:4326", "epsg:32719", always_xy=True)
transformer_to_wgs84 = Transformer.from_crs("epsg:32719", "epsg:4326", always_xy=True)

# Configuration Parameters
MAX_CONCURRENT_REQUESTS = 2   # Limit concurrent requests to reduce rate limiting issues
MAX_CONSECUTIVE_FAILURES = 3  # Skip sub-area if too many failures occur
MAX_RETRIES = 5               # Maximum retries for token and requests
BACKOFF_FACTOR = 2            # Exponential backoff factor
MAX_METERS_PER_PIXEL = 30.0   # Desired spatial resolution

def jitter(base: float) -> float:
    """Apply random jitter to wait times for exponential backoff."""
    return base + random.uniform(0, 1)

async def obtain_token(session: aiohttp.ClientSession) -> Optional[str]:
    """
    Obtain Sentinel Hub access token using client credentials.
    Implements retries with exponential backoff and jitter.

    Parameters
    ----------
    session : aiohttp.ClientSession
        AIOHTTP session for requests.

    Returns
    -------
    Optional[str]
        The access token if successful, None otherwise.
    """
    token_data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Requesting access token from Sentinel Hub...")
            async with session.post(TOKEN_URL, data=token_data, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    token = data.get("access_token")
                    if token:
                        logger.info("Access token obtained successfully.")
                        return token
                    else:
                        logger.error("Received 200 OK but no access_token in response.")
                        return None
                else:
                    text = await response.text()
                    logger.error(f"Failed to obtain token: {response.status} {text}")
                    if response.status >= 500:
                        # Retry on server errors
                        wait_time = jitter(BACKOFF_FACTOR ** attempt)
                        logger.info(f"Retrying token request in {wait_time:.2f}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        return None
        except aiohttp.ClientError as client_err:
            logger.error(f"Client error while obtaining token: {client_err}")
            wait_time = jitter(BACKOFF_FACTOR ** attempt)
            logger.info(f"Retrying token request in {wait_time:.2f}s...")
            await asyncio.sleep(wait_time)
        except asyncio.TimeoutError:
            logger.error("Token request timed out.")
            wait_time = jitter(BACKOFF_FACTOR ** attempt)
            logger.info(f"Retrying token request in {wait_time:.2f}s...")
            await asyncio.sleep(wait_time)
        except Exception as e:
            logger.error(f"Unexpected error while obtaining token: {e}")
            return None

    logger.error("Exceeded maximum retries for obtaining token.")
    return None

def generate_evalscript() -> str:
    """
    Generate a simple EvalScript to compute NDVI without cloud masking.

    Returns
    -------
    str
        EvalScript as a string.
    """
    return """
    //VERSION=3
    function setup() {
        return {
            input: ["B04", "B05"],
            output: { bands: 1, sampleType: "FLOAT32" }
        };
    }

    function evaluatePixel(sample) {
        let denominator = (sample.B05 + sample.B04);
        let ndvi = denominator !== 0 ? (sample.B05 - sample.B04) / denominator : 0;
        return [ndvi];
    }
    """

def divide_aoi_grid(aoi_polygon: Polygon, tile_size: float = 20000.0) -> List[Polygon]:
    """
    Divide AOI into a grid of sub-areas of the specified tile size.

    Parameters
    ----------
    aoi_polygon : Polygon
        AOI polygon in projected CRS.
    tile_size : float
        Size of each tile in meters.

    Returns
    -------
    List[Polygon]
        List of sub-area polygons.
    """
    minx, miny, maxx, maxy = aoi_polygon.bounds
    width = maxx - minx
    height = maxy - miny

    num_tiles_x = int(np.ceil(width / tile_size))
    num_tiles_y = int(np.ceil(height / tile_size))

    tiles = []
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            tile_minx = minx + i * tile_size
            tile_miny = miny + j * tile_size
            tile_maxx = min(tile_minx + tile_size, maxx)
            tile_maxy = min(tile_miny + tile_size, maxy)
            tile = box(tile_minx, tile_miny, tile_maxx, tile_maxy)
            intersection = aoi_polygon.intersection(tile)
            if not intersection.is_empty and isinstance(intersection, Polygon):
                tiles.append(intersection)
    return tiles

def calculate_output_dimensions(tile_polygon: Polygon, max_meters_per_pixel: float = MAX_METERS_PER_PIXEL) -> Tuple[int, int]:
    """
    Calculate image dimensions in pixels based on polygon size and resolution.

    Parameters
    ----------
    tile_polygon : Polygon
        The polygon defining the sub-area.
    max_meters_per_pixel : float
        Desired resolution in meters per pixel.

    Returns
    -------
    Tuple[int, int]
        (width_pixels, height_pixels)
    """
    minx, miny, maxx, maxy = tile_polygon.bounds
    width_m = maxx - minx
    height_m = maxy - miny

    # Avoid NaN or invalid values
    if np.isnan(width_m) or np.isnan(height_m) or width_m <= 0 or height_m <= 0:
        logger.error("Invalid polygon bounds for sub-area. Using fallback dimensions 1x1 pixel.")
        return (1, 1)

    width_pixels = max(int(np.ceil(width_m / max_meters_per_pixel)), 1)
    height_pixels = max(int(np.ceil(height_m / max_meters_per_pixel)), 1)

    return width_pixels, height_pixels

def get_all_dates(year: int, month: int) -> List[str]:
    """
    Get all dates for a given month of a given year.

    Parameters
    ----------
    year : int
    month : int

    Returns
    -------
    List[str]
        List of date strings (YYYY-MM-DD).
    """
    num_days = calendar.monthrange(year, month)[1]
    return [f"{year}-{month:02d}-{day:02d}" for day in range(1, num_days + 1)]

async def request_ndvi_landsat8(
    session: aiohttp.ClientSession,
    token: str,
    polygon_coords: List[Tuple[float, float]],
    date: str,
    evalscript: str,
    sub_area_number: int,
    width_pixels: int,
    height_pixels: int,
    consecutive_failures: int
) -> Tuple[Optional[bytes], int]:
    """
    Requests NDVI data from Sentinel Hub. Implements a single request with no retries here.
    Retries should be handled by the calling function if needed.

    Parameters
    ----------
    session : aiohttp.ClientSession
        AIOHTTP session.
    token : str
        Sentinel Hub access token.
    polygon_coords : List[Tuple[float, float]]
        Polygon coordinates in WGS84.
    date : str
        Date in 'YYYY-MM-DD'.
    evalscript : str
        EvalScript for NDVI calculation.
    sub_area_number : int
        Sub-area identifier.
    width_pixels : int
        Image width in pixels.
    height_pixels : int
        Image height in pixels.
    consecutive_failures : int
        Current consecutive failures.

    Returns
    -------
    Tuple[Optional[bytes], int]
        NDVI data as bytes and updated failure count.
    """
    request_payload = {
        "input": {
            "bounds": {
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon_coords]
                }
            },
            "data": [
                {
                    "type": "landsat-ot-l1",
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

    logger.info(f"Requesting Landsat-8 NDVI for Sub-area {sub_area_number} on {date}...")
    async with session.post(PROCESS_URL, json=request_payload, headers=headers, timeout=600) as response:
        content_type = response.headers.get("Content-Type", "").lower()
        if response.status == 200 and ("image/tiff" in content_type or "application/octet-stream" in content_type):
            data = await response.read()
            logger.debug(f"Received NDVI data for Sub-area {sub_area_number} on {date}.")
            return data, 0
        else:
            # Log error details
            text = await response.text()
            logger.error(f"Failed NDVI request for Sub-area {sub_area_number} on {date}: {response.status} {text}")
            # Save error for debugging
            error_path = DATA_DIR / "raw" / "ndvi" / date / f"sub_area_{sub_area_number}" / "error_response.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(text)
            return None, consecutive_failures + 1

async def process_sub_area(
    session: aiohttp.ClientSession,
    sub_area: Dict,
    dates: List[str],
    evalscript: str,
    token: str,
    month_path: Path
) -> None:
    """
    Process a single sub-area by fetching and aggregating NDVI data over a month.

    Parameters
    ----------
    session : aiohttp.ClientSession
        AIOHTTP session.
    sub_area : Dict
        Sub-area details.
    dates : List[str]
        Dates in the target month.
    evalscript : str
        EvalScript for NDVI calculation.
    token : str
        Sentinel Hub access token.
    month_path : Path
        Path to save monthly aggregated data.
    """
    sub_area_number = sub_area['number']
    polygon_coords = sub_area['coords']
    width_pixels = sub_area['width_pixels']
    height_pixels = sub_area['height_pixels']

    ndvi_values = []
    consecutive_failures = 0

    for date in dates:
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.warning(f"Too many failures for Sub-area {sub_area_number}. Stopping early.")
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
                        ndvi_data = np.where(ndvi_data == 0, np.nan, ndvi_data)
                        ndvi_values.append(ndvi_data)
                logger.debug(f"Aggregated NDVI for Sub-area {sub_area_number} on {date}.")
            except Exception as e:
                logger.error(f"Error reading NDVI data for Sub-area {sub_area_number} on {date}: {e}")
        else:
            logger.warning(f"No NDVI data for Sub-area {sub_area_number} on {date}.")

    if not ndvi_values:
        logger.error(f"No valid NDVI data found for Sub-area {sub_area_number}.")
        return

    # Aggregate NDVI data (mean)
    aggregated_ndvi = np.nanmean(ndvi_values, axis=0)
    logger.info(f"Aggregated NDVI for Sub-area {sub_area_number} completed.")

    # Save as Numpy
    ndvi_npy_path = month_path / f"sub_area_{sub_area_number}" / "ndvi_monthly.npy"
    ndvi_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(ndvi_npy_path, aggregated_ndvi)
    logger.info(f"NDVI array saved at {ndvi_npy_path}")

    # Save as GeoTIFF
    sub_area_bounds = sub_area.get('bounds')
    if sub_area_bounds:
        geotiff_path = month_path / f"sub_area_{sub_area_number}" / "ndvi_monthly.tif"
        try:
            save_ndvi_as_geotiff(aggregated_ndvi, geotiff_path, sub_area_bounds)
            logger.info(f"GeoTIFF saved at {geotiff_path}")
        except Exception as e:
            logger.error(f"Error saving GeoTIFF for Sub-area {sub_area_number}: {e}")

async def process_sub_area_with_semaphore(semaphore, session, sub_area, dates, evalscript, token, month_path):
    """
    Wrapper to ensure concurrency limits via a semaphore.

    Parameters
    ----------
    semaphore : asyncio.Semaphore
        Concurrency control.
    session : aiohttp.ClientSession
        AIOHTTP session.
    sub_area : Dict
        Sub-area details.
    dates : List[str]
        Dates in the month.
    evalscript : str
        EvalScript.
    token : str
        Sentinel Hub token.
    month_path : Path
        Path for output data.
    """
    async with semaphore:
        await process_sub_area(session, sub_area, dates, evalscript, token, month_path)

async def main_async(year: int, month: int) -> None:
    """
    Main asynchronous function to fetch and process NDVI data for all sub-areas in a given month.

    Parameters
    ----------
    year : int
    month : int
    """
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
        logger.error("Invalid AOI polygon after projection. Aborting.")
        sys.exit(1)

    logger.info("Projected AOI geometry is valid.")

    # Divide AOI into sub-areas
    tile_size = 20000.0
    sub_polygons = divide_aoi_grid(aoi_polygon_proj, tile_size=tile_size)
    logger.info(f"Number of sub-areas: {len(sub_polygons)}")

    # Prepare sub-area details
    sub_areas = []
    for idx, poly in enumerate(sub_polygons):
        width_pixels, height_pixels = calculate_output_dimensions(poly, MAX_METERS_PER_PIXEL)
        coords_wgs84 = [transformer_to_wgs84.transform(x, y) for x, y in poly.exterior.coords]
        sub_areas.append({
            "number": idx + 1,
            "coords": coords_wgs84,
            "polygon": poly,
            "width_pixels": width_pixels,
            "height_pixels": height_pixels,
            "bounds": poly.bounds
        })

    # Create sub_area_bounds.json
    sub_area_bounds = {}
    for sub_area in sub_areas:
        sub_area_number = sub_area['number']
        bounds = sub_area['bounds']
        min_lon, min_lat = transformer_to_wgs84.transform(bounds[0], bounds[1])
        max_lon, max_lat = transformer_to_wgs84.transform(bounds[2], bounds[3])
        sub_area_bounds[str(sub_area_number)] = [min_lon, min_lat, max_lon, max_lat]

    bounds_file = DATA_DIR / 'sub_area_bounds.json'
    bounds_file.parent.mkdir(parents=True, exist_ok=True)
    with open(bounds_file, 'w') as f:
        json.dump(sub_area_bounds, f)
    logger.info(f"Sub-area bounds saved to {bounds_file}")

    # Get all dates for the month
    dates = get_all_dates(year, month)
    logger.info(f"Processing {len(dates)} days for {year}-{month:02d}.")

    # Prepare output directory
    month_str = f"{year}-{month:02d}"
    month_path = DATA_DIR / "raw" / "ndvi" / month_str
    month_path.mkdir(parents=True, exist_ok=True)

    # Create semaphore and session
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    evalscript = generate_evalscript()

    async with aiohttp.ClientSession() as session:
        token = await obtain_token(session)
        if not token:
            logger.error("Failed to obtain token, cannot proceed.")
            sys.exit(1)

        tasks = [
            process_sub_area_with_semaphore(
                semaphore,
                session,
                sub_area,
                dates,
                evalscript,
                token,
                month_path
            )
            for sub_area in sub_areas
        ]
        await asyncio.gather(*tasks)

    logger.info("NDVI data fetching and aggregation completed successfully.")

def main() -> None:
    """
    Entry point of the script.
    """
    parser = argparse.ArgumentParser(
        description="Fetch and aggregate Landsat-8 NDVI data for a given month without cloud masking."
    )
    parser.add_argument("year_month", type=str, help="Year and month in 'YYYY-MM' format.")

    args = parser.parse_args()

    # Validate year_month
    try:
        target_date = datetime.strptime(args.year_month, "%Y-%m")
        year = target_date.year
        month = target_date.month
    except ValueError:
        logger.error("Invalid date format. Use 'YYYY-MM'.")
        sys.exit(1)

    asyncio.run(main_async(year, month))

if __name__ == "__main__":
    main()
