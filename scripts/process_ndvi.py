#!/usr/bin/env python3
"""
Process Aggregated Seasonal NDVI Data to Compute Statistical Metrics and Prepare for Analysis

This script processes aggregated seasonal NDVI data to compute statistical metrics,
ensuring the data is ready for further analysis or visualization.

Usage:
    python scripts/process_ndvi.py <YEAR> <SEASON> [--sub_areas SUB_AREA_NUMBERS] [--method <mean|median|max>]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
from loguru import logger

# Adjust the import path based on your project structure
# Assuming 'utils' is a subpackage within 'scripts'
from utils.utils import (
    aggregate_seasonal_ndvi,
    select_best_images,
    save_ndvi_as_geotiff,
    load_seasonal_ndvi,
)


# Configure Logging
def configure_logging(log_file: Path, log_level: str = "DEBUG"):
    """
    Configure logging with loguru.

    Parameters
    ----------
    log_file : Path
        Path to the log file.
    log_level : str, optional
        Logging level, by default "DEBUG".
    """
    logger.remove()  # Remove default logger to prevent duplication
    logger.add(log_file, rotation="1 MB", retention="7 days", level=log_level)
    logger.add(sys.stdout, level="INFO")  # Add console logging


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


def main():
    """
    Entry point of the process_ndvi.py script.
    """
    parser = argparse.ArgumentParser(
        description="Process aggregated seasonal NDVI data to compute statistical metrics."
    )
    parser.add_argument(
        "year",
        type=int,
        help="Target year as an integer (e.g., 2020)."
    )
    parser.add_argument(
        "season",
        type=str,
        choices=["spring", "summer", "autumn", "winter"],
        help="Season to process (e.g., spring)."
    )
    parser.add_argument(
        "--sub_areas",
        type=int,
        nargs='+',
        help="Sub-area number(s) to process. If not specified, all sub-areas will be processed."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["mean", "median", "max"],
        default="mean",
        help="Aggregation method used during fetching."
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
        help="Path to save the statistics JSON files."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="DEBUG",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)."
    )

    args = parser.parse_args()

    # Configure Logging
    log_file = Path("logs") / "process_ndvi.log"
    configure_logging(log_file, log_level=args.log_level)

    # Validate year and season
    if args.year <= 0:
        logger.error("Invalid year provided.")
        sys.exit(1)

    SEASONS = {
        "spring": [9, 10, 11],   # September, October, November
        "summer": [12, 1, 2],    # December, January, February
        "autumn": [3, 4, 5],     # March, April, May
        "winter": [6, 7, 8],     # June, July, August
    }

    season = args.season.lower()
    months = SEASONS.get(season)
    if not months:
        logger.error(f"Invalid season: {season}")
        sys.exit(1)

    # Handle seasons that span two years (e.g., winter)
    if season == "winter":
        months_with_years = []
        for m in months:
            if m == 12:
                months_with_years.append((args.year - 1, m))
            else:
                months_with_years.append((args.year, m))
    else:
        months_with_years = [(args.year, m) for m in months]

    # Compile month strings in 'YYYY-MM' format
    season_months = [f"{y}-{m:02d}" for y, m in months_with_years]
    logger.info(f"Season '{season}' months: {season_months}")

    # Define path to save seasonal statistics (ensure directories exist)
    for month_str in season_months:
        month_path = Path(args.data_dir) / "raw" / "ndvi" / month_str
        month_path.mkdir(parents=True, exist_ok=True)

    # Load sub-area bounds
    bounds_file = Path(args.data_dir) / 'sub_area_bounds.json'
    if not bounds_file.exists():
        logger.error(f"Sub-area bounds file not found: {bounds_file}")
        sys.exit(1)
    with open(bounds_file, 'r') as f:
        sub_area_bounds = json.load(f)

    # Determine sub-areas to process
    if args.sub_areas:
        sub_areas = args.sub_areas
    else:
        # If not specified, process all sub-areas present in sub_area_bounds
        sub_areas = [int(k) for k in sub_area_bounds.keys()]
    logger.info(f"Processing Sub-areas: {sub_areas}")

    # Process each sub-area
    for sub_area_number in sub_areas:
        logger.info(f"Processing Sub-area {sub_area_number}")

        # Load NDVI data for all months in the season
        seasonal_ndvi_list = load_seasonal_ndvi(season_months, sub_area_number, Path(args.data_dir))

        # Select the best images based on valid pixels
        best_images = select_best_images(seasonal_ndvi_list, top_n=3)  # You can adjust top_n if needed

        if not best_images:
            logger.error(f"No valid NDVI images found for Sub-area {sub_area_number}. Skipping.")
            continue

        # Aggregate the best images
        aggregated_seasonal_ndvi = aggregate_seasonal_ndvi(best_images, method=args.method)

        if aggregated_seasonal_ndvi.size == 0:
            logger.error(f"Aggregated NDVI data invalid for Sub-area {sub_area_number}. Skipping.")
            continue

        # Compute statistical metrics
        statistics = compute_statistics(aggregated_seasonal_ndvi)
        logger.debug(f"Computed statistics: {statistics}")

        # Define output path for statistics
        stats_filename = f"ndvi_{season}_{args.year}_sub_area_{sub_area_number}.json"
        stats_path = Path(args.output_dir) / stats_filename
        save_statistics(statistics, stats_path)

        # Optionally, save aggregated NDVI as GeoTIFF
        bounds = sub_area_bounds.get(str(sub_area_number))
        if bounds:
            geotiff_filename = f"ndvi_{season}_{args.year}_sub_area_{sub_area_number}.tif"
            geotiff_path = Path(args.output_dir) / geotiff_filename
            try:
                save_ndvi_as_geotiff(aggregated_seasonal_ndvi, geotiff_path, bounds)
                logger.info(f"Aggregated seasonal NDVI GeoTIFF saved to {geotiff_path}")
            except Exception as e:
                logger.error(f"Error saving aggregated seasonal NDVI GeoTIFF for Sub-area {sub_area_number}: {e}")
        else:
            logger.warning(f"Bounds not found for Sub-area {sub_area_number}. Skipping GeoTIFF save.")
