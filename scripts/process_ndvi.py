#!/usr/bin/env python3
"""
Process Aggregated NDVI Data to Compute Statistical Metrics

This script processes the aggregated monthly NDVI data to compute statistical
metrics, identify trends, and prepare the data for further analysis.

Usage:
    python scripts/process_ndvi.py YYYY-MM [--sub_area SUB_AREA_NUMBER]
"""
import argparse
import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from loguru import logger

def load_ndvi_data(data_dir: Path, month: str, sub_area: Optional[int] = None) -> np.ndarray:
    """
    Load aggregated NDVI data for a specific month and sub-area.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory.
    month : str
        Target month in 'YYYY-MM' format.
    sub_area : Optional[int], optional
        Sub-area number, by default None (loads all sub-areas).

    Returns
    -------
    np.ndarray
        NDVI data array.
    """
    ndvi_path = data_dir / "raw" / "ndvi" / month
    if sub_area:
        ndvi_file = ndvi_path / f"sub_area_{sub_area}" / "ndvi_monthly.npy"
        if not ndvi_file.exists():
            logger.error(f"NDVI file not found: {ndvi_file}")
            sys.exit(1)
        return np.load(ndvi_file)
    else:
        # If no sub_area is specified, return a list of all sub-area NDVI arrays
        sub_area_files = list(ndvi_path.glob("sub_area_*/ndvi_monthly.npy"))
        if not sub_area_files:
            logger.error(f"No NDVI data found for month {month}.")
            sys.exit(1)
        ndvi_data_list = []
        for idx, f in enumerate(sub_area_files, start=1):
            ndvi = np.load(f)
            logger.debug(f"Sub-area {idx} NDVI shape: {ndvi.shape}")
            ndvi_data_list.append(ndvi)
        return ndvi_data_list

def compute_statistics(ndvi_data: np.ndarray) -> Dict[str, float]:
    """
    Compute statistical metrics for NDVI data.

    Parameters
    ----------
    ndvi_data : np.ndarray
        NDVI data array.

    Returns
    -------
    Dict[str, float]
        Dictionary containing statistical metrics.
    """
    stats = {
        "mean_ndvi": np.nanmean(ndvi_data),
        "median_ndvi": np.nanmedian(ndvi_data),
        "std_ndvi": np.nanstd(ndvi_data),
        "min_ndvi": np.nanmin(ndvi_data),
        "max_ndvi": np.nanmax(ndvi_data),
        "count_valid_pixels": np.count_nonzero(~np.isnan(ndvi_data))
    }
    return stats

def save_statistics(
    stats: Dict[str, float],
    output_path: Path
) -> None:
    """
    Save statistical metrics to a JSON file.

    Parameters
    ----------
    stats : Dict[str, float]
        Statistical metrics dictionary.
    output_path : Path
        Path to save the statistics JSON file.
    """
    # Convert all NumPy float types to native Python float
    stats = {k: float(v) for k, v in stats.items()}

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Statistics saved to {output_path}")

def main():
    """
    Entry point of the process_ndvi.py script.
    """
    parser = argparse.ArgumentParser(
        description="Process aggregated NDVI data to compute statistical metrics."
    )
    parser.add_argument(
        "month",
        type=str,
        help="Target month for processing in YYYY-MM format."
    )
    parser.add_argument(
        "--sub_area",
        type=int,
        default=None,
        help="Specific sub-area number to process. If not provided, statistics will be computed for each sub-area individually."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the statistics JSON file. If not provided, it will be saved in the results directory."
    )

    args = parser.parse_args()

    # Define project directories
    SCRIPT_DIR = Path(__file__).resolve().parent  # scripts/
    PROJECT_ROOT = SCRIPT_DIR.parent  # proyecto_ndvi/
    DATA_DIR = PROJECT_ROOT / 'data'  # proyecto_ndvi/data/
    RESULTS_DIR = PROJECT_ROOT / 'results' / 'statistics'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.sub_area:
        # Process a specific sub-area
        ndvi_data = load_ndvi_data(DATA_DIR, args.month, args.sub_area)
        # Compute statistics
        stats = compute_statistics(ndvi_data)
        # Define output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_filename = f"ndvi_statistics_{args.month}_sub_area_{args.sub_area}.json"
            output_path = RESULTS_DIR / output_filename
        # Save statistics
        save_statistics(stats, output_path)
    else:
        # Process all sub-areas individually
        ndvi_data_list = load_ndvi_data(DATA_DIR, args.month)
        for idx, ndvi_data in enumerate(ndvi_data_list, start=1):
            stats = compute_statistics(ndvi_data)
            if args.output:
                output_path = Path(args.output).parent / f"ndvi_statistics_{args.month}_sub_area_{idx}.json"
            else:
                output_filename = f"ndvi_statistics_{args.month}_sub_area_{idx}.json"
                output_path = RESULTS_DIR / output_filename
            save_statistics(stats, output_path)

if __name__ == "__main__":
    main()
