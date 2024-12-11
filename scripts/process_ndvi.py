#!/usr/bin/env python3
"""
Process Aggregated Monthly NDVI Data to Compute Statistical Metrics

This script processes aggregated monthly NDVI data to:
- Select the best images (with most valid pixels),
- Aggregate them using a specified method (mean, median, max),
- Compute statistical metrics (mean, median, std, min, max),
- Save results as JSON and optionally as GeoTIFF.

Usage:
    python scripts/process_ndvi.py <YEAR> <MONTH> [--sub_areas SUB_AREA_NUMBERS] [--method <mean|median|max>]

Example:
    python scripts/process_ndvi.py 2020 03 --sub_areas 1 2 3 --method median
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
from loguru import logger

# Importing utility functions (adjust the path if needed)
from scripts.utils.utils import (
    aggregate_monthly_ndvi,
    select_best_images,
    save_ndvi_as_geotiff,
    load_monthly_ndvi_files,
    configure_logging
)

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
    statistics = {
        "mean_ndvi": float(np.nanmean(ndvi)),
        "median_ndvi": float(np.nanmedian(ndvi)),
        "max_ndvi": float(np.nanmax(ndvi)),
        "min_ndvi": float(np.nanmin(ndvi)),
        "std_dev_ndvi": float(np.nanstd(ndvi))
    }
    return statistics

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
    Process NDVI data for a given month and sub-areas, compute statistics, and save results.
    """
    parser = argparse.ArgumentParser(
        description="Process aggregated monthly NDVI data to compute statistical metrics."
    )
    parser.add_argument(
        "year",
        type=int,
        help="Target year as an integer (e.g., 2020)."
    )
    parser.add_argument(
        "month",
        type=int,
        choices=range(1,13),
        help="Target month as an integer (1-12)."
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
        help="Aggregation method to use for NDVI."
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

    # Validate year and month
    if args.year <= 0:
        logger.error("Invalid year provided.")
        sys.exit(1)
    if not 1 <= args.month <= 12:
        logger.error("Invalid month provided. Must be between 1 and 12.")
        sys.exit(1)

    month_str = f"{args.year}-{args.month:02d}"
    logger.info(f"Processing NDVI statistics for {month_str}, Sub-areas: {args.sub_areas if args.sub_areas else 'All'}.")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    bounds_file = data_dir / 'sub_area_bounds.json'
    if not bounds_file.exists():
        logger.error(f"Sub-area bounds file not found: {bounds_file}")
        sys.exit(1)

    with open(bounds_file, 'r') as f:
        sub_area_bounds = json.load(f)

    # Determine sub-areas to process
    if args.sub_areas:
        sub_areas = args.sub_areas
    else:
        sub_areas = [int(k) for k in sub_area_bounds.keys()]
    logger.info(f"Processing Sub-areas: {sub_areas}")

    # Process each sub-area
    for sub_area_number in sub_areas:
        logger.info(f"Processing Sub-area {sub_area_number}")

        # Load NDVI data for the month (raw NDVI .npy files)
        ndvi_list = load_monthly_ndvi_files(args.month, [args.year], sub_area_number, data_dir)
        if not ndvi_list:
            logger.warning(f"No NDVI data found for Sub-area {sub_area_number} in {month_str}. Skipping.")
            continue

        # Select best images based on valid pixels
        best_images = select_best_images(ndvi_list, top_n=3)
        if not best_images:
            logger.warning(f"No valid NDVI images selected for Sub-area {sub_area_number}. Skipping.")
            continue

        # Aggregate the best images
        aggregated_monthly_ndvi = aggregate_monthly_ndvi(best_images, method=args.method)
        if aggregated_monthly_ndvi.size == 0:
            logger.error(f"Aggregated NDVI data invalid for Sub-area {sub_area_number}. Skipping.")
            continue

        # Compute statistics
        statistics = compute_statistics(aggregated_monthly_ndvi)
        logger.debug(f"Computed statistics for Sub-area {sub_area_number}: {statistics}")

        # Save statistics
        stats_filename = f"ndvi_{month_str}_sub_area_{sub_area_number}.json"
        stats_path = output_dir / stats_filename
        save_statistics(statistics, stats_path)

        # Save aggregated NDVI as GeoTIFF if bounds are available
        bounds = sub_area_bounds.get(str(sub_area_number))
        if bounds and 'bounds_proj' in bounds:
            geotiff_filename = f"ndvi_{month_str}_sub_area_{sub_area_number}.tif"
            geotiff_path = output_dir / geotiff_filename
            try:
                save_ndvi_as_geotiff(aggregated_monthly_ndvi, geotiff_path, bounds['bounds_proj'], crs_epsg=32719)
                logger.info(f"Aggregated monthly NDVI GeoTIFF saved to {geotiff_path}")
            except Exception as e:
                logger.error(f"Error saving aggregated monthly NDVI GeoTIFF for Sub-area {sub_area_number}: {e}")
        else:
            logger.warning(f"Projected bounds not found for Sub-area {sub_area_number}. Skipping GeoTIFF save.")

if __name__ == "__main__":
    main()
