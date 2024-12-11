#!/usr/bin/env python3
"""
Process Raw Monthly NDVI Data

This script processes raw NDVI data for a specific month:
- Loads NDVI data for all sub-areas,
- Computes statistical metrics (mean, median, std, min, max),
- Saves the results as JSON and optionally as GeoTIFF.

Usage:
    python scripts/process_ndvi.py YYYY-MM

Example:
    python scripts/process_ndvi.py 2019-03
"""

import sys
import json
from pathlib import Path
import numpy as np
import rasterio
import logging

# Configure directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
PROCESSED_DIR = DATA_DIR / "processed" / "ndvi"
RAW_DIR = DATA_DIR / "raw" / "ndvi"
BOUNDS_FILE = DATA_DIR / "sub_area_bounds.json"

# Configure logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "process_ndvi.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

def load_ndvi_files(month_dir: Path, sub_area: str) -> list:
    """
    Load NDVI files for a specific sub-area.

    Parameters
    ----------
    month_dir : Path
        Path to the directory containing NDVI data for the month.
    sub_area : str
        Sub-area folder name (e.g., 'sub_area_1').

    Returns
    -------
    list
        List of NDVI arrays as NumPy arrays.
    """
    sub_area_dir = month_dir / f"sub_area_{sub_area}"  # Cambiar el formato aquí
    if not sub_area_dir.exists():
        logging.warning(f"Sub-area directory not found: {sub_area_dir}")
        return []

    ndvi_files = list(sub_area_dir.glob("*.npy"))
    ndvi_arrays = []

    for file in ndvi_files:
        try:
            ndvi_array = np.load(file)
            ndvi_arrays.append(ndvi_array)
            logging.info(f"Loaded NDVI file: {file}")
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")

    return ndvi_arrays

def compute_statistics(ndvi_array: np.ndarray) -> dict:
    """
    Compute basic statistics for an NDVI array.

    Parameters
    ----------
    ndvi_array : np.ndarray
        NDVI data as a NumPy array.

    Returns
    -------
    dict
        Dictionary containing statistical metrics.
    """
    return {
        "mean": float(np.nanmean(ndvi_array)),
        "median": float(np.nanmedian(ndvi_array)),
        "std_dev": float(np.nanstd(ndvi_array)),
        "min": float(np.nanmin(ndvi_array)),
        "max": float(np.nanmax(ndvi_array)),
    }

def process_month(year_month: str):
    """
    Process NDVI data for all sub-areas in a specific month.

    Parameters
    ----------
    year_month : str
        The month to process in 'YYYY-MM' format.
    """
    # Ensure data directory exists
    month_dir = RAW_DIR / year_month
    if not month_dir.exists():
        logging.error(f"Month directory not found: {month_dir}")
        sys.exit(1)

    # Ensure bounds file exists
    if not BOUNDS_FILE.exists():
        logging.error(f"Bounds file not found: {BOUNDS_FILE}")
        sys.exit(1)

    with open(BOUNDS_FILE, "r") as f:
        sub_area_bounds = json.load(f)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = PROCESSED_DIR / year_month
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each sub-area
    for sub_area in sub_area_bounds.keys():
        logging.info(f"Processing Sub-area: {sub_area}")

        # Load NDVI data
        ndvi_arrays = load_ndvi_files(month_dir, sub_area)
        if not ndvi_arrays:
            logging.warning(f"No NDVI data found for Sub-area {sub_area}. Skipping.")
            continue

        # Aggregate NDVI data (mean)
        aggregated_ndvi = np.nanmean(ndvi_arrays, axis=0)

        # Compute statistics
        stats = compute_statistics(aggregated_ndvi)
        logging.info(f"Computed statistics for Sub-area {sub_area}: {stats}")

        # Save statistics to JSON
        stats_file = output_dir / f"ndvi_statistics_{sub_area}.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=4)
        logging.info(f"Statistics saved: {stats_file}")

        # Save aggregated NDVI as GeoTIFF
        ndvi_tif = output_dir / f"ndvi_{sub_area}.tif"
        ref_tif_path = month_dir / sub_area / "ndvi_monthly.tif"

        if ref_tif_path.exists():
            try:
                with rasterio.open(ref_tif_path) as ref_src:
                    profile = ref_src.profile.copy()
                    profile.update(dtype="float32", count=1, nodata=np.nan)

                    with rasterio.open(ndvi_tif, "w", **profile) as dst:
                        dst.write(aggregated_ndvi, 1)
                logging.info(f"GeoTIFF saved: {ndvi_tif}")
            except Exception as e:
                logging.error(f"Error saving GeoTIFF for Sub-area {sub_area}: {e}")
        else:
            logging.warning(f"Reference GeoTIFF not found for Sub-area {sub_area}. Skipping GeoTIFF save.")

def main():
    """
    Main entry point for the script.
    """
    if len(sys.argv) < 2:
        logging.error("Usage: python scripts/process_ndvi.py YYYY-MM")
        sys.exit(1)

    year_month = sys.argv[1]
    process_month(year_month)

if __name__ == "__main__":
    main()
