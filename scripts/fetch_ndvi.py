#!/usr/bin/env python3
"""
Fetch NDVI Data for a Specific Month without Cloud Masking

This script downloads Landsat-8 NDVI data for all days of a specific month,
aggregates the valid NDVI data to form a monthly representation without applying a cloud mask.

Usage:
    python scripts/fetch_ndvi.py YYYY-MM
"""

import asyncio
import aiohttp
import os
import sys
import json
import calendar
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import rasterio
from loguru import logger
from shapely.geometry import Polygon, box
from pyproj import Transformer
from dotenv import load_dotenv

from utils import save_ndvi_as_geotiff  # Ensure this import from utils.py

# Load environment variables from .env file
load_dotenv()

# Define project directories
SCRIPT_DIR = Path(__file__).resolve().parent  # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent  # proyecto_ndvi/
DATA_DIR = PROJECT_ROOT / 'data'  # proyecto_ndvi/data/
LOGS_DIR = PROJECT_ROOT / 'logs'  # proyecto_ndvi/logs/

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure Logging
LOG_FILE = LOGS_DIR / 'fetch_ndvi.log'
LOG_LEVEL = os.getenv("FETCH_NDVI_LOG_LEVEL", "DEBUG").upper()

logger.remove()  # Remove default logger to prevent duplication
logger.add(LOG_FILE, rotation="1 MB", retention="7 days", level=LOG_LEVEL)
logger.add(sys.stdout, level=LOG_LEVEL)  # Add console logging

# Sentinel Hub Configuration
CLIENT_ID = os.getenv("SENTINELHUB_CLIENT_ID")
CLIENT_SECRET = os.getenv("SENTINELHUB_CLIENT_SECRET")
TOKEN_URL = os.getenv("SENTINELHUB_TOKEN_URL", "https://services.sentinel-hub.com/oauth/token")
PROCESS_URL = os.getenv("SENTINELHUB_PROCESS_URL", "https://services-uswest2.sentinel-hub.com/api/v1/process")

# Verify Credentials
if not CLIENT_ID or not CLIENT_SECRET:
    logger.error("Environment variables SENTINELHUB_CLIENT_ID and SENTINELHUB_CLIENT_SECRET must be set.")
    sys.exit(1)

# CRS Transformers
transformer_to_proj = Transformer.from_crs("epsg:4326", "epsg:32719", always_xy=True)
transformer_to_wgs84 = Transformer.from_crs("epsg:32719", "epsg:4326", always_xy=True)

# Configuration Parameters
MAX_CONCURRENT_REQUESTS = 2  # Further reduced to minimize rate limiting
MAX_CONSECUTIVE_FAILURES = 3  # Reduced to skip quickly if no data
BACKOFF_FACTOR = 2  # Exponential backoff factor

async def obtain_token(session: aiohttp.ClientSession) -> Optional[str]:
    token_data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    max_retries = 5

    for attempt in range(max_retries):
        try:
            logger.info("Requesting access token from Sentinel Hub...")
            async with session.post(TOKEN_URL, data=token_data, headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("Access token obtained successfully.")
                    return data.get("access_token")
                else:
                    text = await response.text()
                    logger.error(f"Failed to obtain token: {response.status} {text}")
                    if response.status >= 500:
                        # Retry on server errors
                        await asyncio.sleep(BACKOFF_FACTOR ** attempt)
                        continue
                    else:
                        return None
        except aiohttp.ClientError as client_err:
            logger.error(f"Client error while obtaining token: {client_err}")
            await asyncio.sleep(BACKOFF_FACTOR ** attempt)
        except asyncio.TimeoutError:
            logger.error("Token request timed out.")
            await asyncio.sleep(BACKOFF_FACTOR ** attempt)
        except Exception as e:
            logger.error(f"Unexpected error while obtaining token: {e}")
            return None
    logger.error("Exceeded maximum retries for obtaining token.")
    return None

def generate_evalscript() -> str:
    """
    Generates the EvalScript for calculating NDVI without cloud masking.

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
    Divides the AOI into a grid of sub-areas of the specified size.

    Parameters
    ----------
    aoi_polygon : Polygon
        Shapely Polygon object representing the AOI in projected CRS.
    tile_size : float, optional
        Size of each sub-area in meters, by default 20000.0 (20 km).

    Returns
    -------
    List[Polygon]
        List of Shapely Polygon objects representing sub-areas.
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

def calculate_output_dimensions(tile_polygon: Polygon, max_meters_per_pixel: float = 30.0) -> Tuple[int, int]:
    """
    Calculates the output width and height in pixels for the given sub-area.

    Parameters
    ----------
    tile_polygon : Polygon
        Shapely Polygon object representing the sub-area in projected CRS.
    max_meters_per_pixel : float, optional
        Maximum meters per pixel, by default 30.0.

    Returns
    -------
    Tuple[int, int]
        Tuple containing (width_pixels, height_pixels).
    """
    minx, miny, maxx, maxy = tile_polygon.bounds
    width_meters = maxx - minx
    height_meters = maxy - miny

    width_pixels = max(int(np.ceil(width_meters / max_meters_per_pixel)), 1)
    height_pixels = max(int(np.ceil(height_meters / max_meters_per_pixel)), 1)

    return width_pixels, height_pixels

def get_all_dates(year: int, month: int) -> List[str]:
    """
    Retrieves all dates for a specific year and month.

    Parameters
    ----------
    year : int
        Year as an integer.
    month : int
        Month as an integer.

    Returns
    -------
    List[str]
        List of date strings in YYYY-MM-DD format.
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
    Requests Landsat-8 NDVI data from Sentinel Hub for a specific sub-area and date.

    Parameters
    ----------
    session : aiohttp.ClientSession
        aiohttp session object.
    token : str
        Sentinel Hub access token.
    polygon_coords : List[Tuple[float, float]]
        List of coordinates defining the sub-area polygon.
    date : str
        Target date in YYYY-MM-DD format.
    evalscript : str
        EvalScript for calculating NDVI without cloud masking.
    sub_area_number : int
        Identifier for the sub-area.
    width_pixels : int
        Output image width in pixels.
    height_pixels : int
        Output image height in pixels.
    consecutive_failures : int
        Current count of consecutive failed requests.

    Returns
    -------
    Tuple[Optional[bytes], int]
        NDVI GeoTIFF data as bytes or None if the request fails, and updated consecutive_failures count.
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
                    "type": "landsat-ot-l1",  # Correct data collection type
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
        logger.info(f"Requesting Landsat-8 NDVI for sub-area {sub_area_number} on {date}...")
        async with session.post(PROCESS_URL, json=request_payload, headers=headers, timeout=600) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if response.status == 200 and ("image/tiff" in content_type or "application/octet-stream" in content_type):
                content = await response.read()
                logger.debug(f"Received NDVI data for sub-area {sub_area_number} on {date}.")
                return content, 0  # Reset failure count on success
            elif response.status == 429:
                # Handle rate limiting
                logger.warning(f"Rate limit exceeded for sub-area {sub_area_number} on {date}.")
                return None, consecutive_failures + 1
            elif response.status >= 500:
                # Server error, may retry
                logger.warning(f"Server error {response.status} for sub-area {sub_area_number} on {date}.")
                return None, consecutive_failures + 1
            else:
                text = await response.text()
                logger.error(f"Failed request for Landsat-8 NDVI for sub-area {sub_area_number} on {date}: {response.status} {text}")
                # Save error response for debugging
                error_path = DATA_DIR / "raw" / "ndvi" / date / f"sub_area_{sub_area_number}" / "error_response.txt"
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(text)
                return None, consecutive_failures + 1
    except asyncio.TimeoutError:
        logger.error(f"Landsat-8 request timed out for sub-area {sub_area_number} on {date}.")
        return None, consecutive_failures + 1
    except aiohttp.ClientError as client_err:
        logger.error(f"Client error during Landsat-8 request for sub-area {sub_area_number} on {date}: {client_err}.")
        return None, consecutive_failures + 1
    except Exception as e:
        logger.error(f"Unexpected error during Landsat-8 request for sub-area {sub_area_number} on {date}: {e}")
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
    Processes a specific sub-area by requesting NDVI data for all valid days of the month,
    aggregating the valid NDVI data to form a monthly representation without cloud masking.

    Parameters
    ----------
    session : aiohttp.ClientSession
        aiohttp session object.
    sub_area : Dict
        Dictionary containing sub-area details.
    dates : List[str]
        List of date strings in YYYY-MM-DD format.
    evalscript : str
        EvalScript for calculating NDVI without cloud masking.
    token : str
        Sentinel Hub access token.
    month_path : Path
        Path to save the aggregated monthly NDVI data.
    """
    sub_area_number = sub_area['number']
    polygon_coords = sub_area['coords']
    width_pixels = sub_area['width_pixels']
    height_pixels = sub_area['height_pixels']

    ndvi_values = []
    consecutive_failures = 0  # Initialize failure counter

    for date in dates:
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.warning(f"Maximum consecutive failures reached for sub-area {sub_area_number}. Skipping remaining dates.")
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
                        # Replace noDataValue with np.nan for aggregation
                        ndvi_data = np.where(ndvi_data == 0, np.nan, ndvi_data)
                        ndvi_values.append(ndvi_data)
                logger.debug(f"NDVI data for sub-area {sub_area_number} on {date} aggregated for monthly aggregation.")
            except Exception as e:
                logger.error(f"Error processing NDVI data for sub-area {sub_area_number} on {date}: {e}")
        else:
            logger.warning(f"No NDVI data obtained for sub-area {sub_area_number} on {date}.")

    if not ndvi_values:
        logger.error(f"No valid NDVI data found for sub-area {sub_area_number} in the month.")
        return

    # Aggregate NDVI data (e.g., mean)
    aggregated_ndvi = np.nanmean(ndvi_values, axis=0)
    logger.info(f"Aggregated NDVI data for sub-area {sub_area_number}.")

    # Save aggregated NDVI as `.npy`
    ndvi_npy_path = month_path / f"sub_area_{sub_area_number}" / "ndvi_monthly.npy"
    try:
        ndvi_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(ndvi_npy_path, aggregated_ndvi)
        logger.info(f"Aggregated monthly NDVI saved to {ndvi_npy_path}")
    except IOError as io_err:
        logger.error(f"Error saving aggregated NDVI for sub-area {sub_area_number}: {io_err}")

    # Optionally, save aggregated NDVI as GeoTIFF
    sub_area_bounds = sub_area.get('bounds')
    if sub_area_bounds:
        geotiff_path = month_path / f"sub_area_{sub_area_number}" / "ndvi_monthly.tif"
        try:
            save_ndvi_as_geotiff(aggregated_ndvi, geotiff_path, sub_area_bounds)
        except Exception as e:
            logger.error(f"Error saving aggregated monthly NDVI GeoTIFF for sub-area {sub_area_number}: {e}")

async def process_sub_area_with_semaphore(semaphore, session, sub_area, dates, evalscript, token, month_path):
    """
    Wrapper to process a sub-area with semaphore control.

    Parameters
    ----------
    semaphore : asyncio.Semaphore
        Semaphore object to limit concurrent requests.
    session : aiohttp.ClientSession
        aiohttp session object.
    sub_area : Dict
        Dictionary containing sub-area details.
    dates : List[str]
        List of date strings in YYYY-MM-DD format.
    evalscript : str
        EvalScript for calculating NDVI without cloud masking.
    token : str
        Sentinel Hub access token.
    month_path : Path
        Path to save the aggregated monthly NDVI data.
    """
    async with semaphore:
        await process_sub_area(session, sub_area, dates, evalscript, token, month_path)

async def main_async(year: int, month: int) -> None:
    """
    Asynchronous main function to process all sub-areas for a given month.

    Parameters
    ----------
    year : int
        Year as an integer.
    month : int
        Month as an integer.
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
        logger.error("Projected AOI geometry is invalid.")
        sys.exit(1)
    else:
        logger.info("Projected AOI geometry is valid.")

    # Divide AOI into sub-areas using grid
    tile_size = 20000.0  # Increased tile size to reduce number of sub-areas
    sub_polygons = divide_aoi_grid(aoi_polygon_proj, tile_size=tile_size)
    logger.info(f"Number of sub-areas: {len(sub_polygons)}")

    # Prepare sub-areas with identifiers and output dimensions
    sub_areas = []
    for idx, poly in enumerate(sub_polygons):
        width_pixels, height_pixels = calculate_output_dimensions(poly)
        coords_wgs84 = [transformer_to_wgs84.transform(x, y) for x, y in poly.exterior.coords]
        sub_areas.append({
            "number": idx + 1,
            "coords": coords_wgs84,
            "polygon": poly,
            "width_pixels": width_pixels,
            "height_pixels": height_pixels,
            "bounds": poly.bounds  # (minx, miny, maxx, maxy) in projected CRS
        })

    # Create a dictionary to store the bounds of each sub-area
    sub_area_bounds = {}
    for sub_area in sub_areas:
        sub_area_number = sub_area['number']
        bounds = sub_area['bounds']
        # Convert projected bounds to WGS84
        min_lon, min_lat = transformer_to_wgs84.transform(bounds[0], bounds[1])
        max_lon, max_lat = transformer_to_wgs84.transform(bounds[2], bounds[3])
        sub_area_bounds[str(sub_area_number)] = [min_lon, min_lat, max_lon, max_lat]

    # Save bounds to a JSON file
    bounds_file = DATA_DIR / 'sub_area_bounds.json'
    bounds_file.parent.mkdir(parents=True, exist_ok=True)
    with open(bounds_file, 'w') as f:
        json.dump(sub_area_bounds, f)
    logger.info(f"Sub-area bounds saved to {bounds_file}")

    # Get all dates for the month
    dates = get_all_dates(year, month)
    logger.info(f"Total days to process for {year}-{month:02d}: {len(dates)}")

    # Define path to save monthly data
    month_str = f"{year}-{month:02d}"
    month_path = DATA_DIR / "raw" / "ndvi" / month_str
    month_path.mkdir(parents=True, exist_ok=True)

    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Generate the EvalScript without cloud masking
    evalscript = generate_evalscript()

    async with aiohttp.ClientSession() as session:
        token = await obtain_token(session)
        if not token:
            logger.error("Failed to obtain Sentinel Hub access token.")
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

    logger.info("NDVI data fetching and aggregation process completed successfully.")

def main() -> None:
    """
    Entry point of the script.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and aggregate Landsat-8 NDVI data for a specific month without cloud masking.")
    parser.add_argument("year_month", type=str, help="Target year and month in YYYY-MM format.")
    args = parser.parse_args()

    # Validate year_month format
    try:
        target_date = datetime.strptime(args.year_month, "%Y-%m")
        year = target_date.year
        month = target_date.month
    except ValueError:
        logger.error("Invalid year-month format. Use YYYY-MM.")
        sys.exit(1)

    asyncio.run(main_async(year, month))

if __name__ == "__main__":
    main()
