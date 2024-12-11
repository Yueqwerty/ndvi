#!/usr/bin/env python3
"""
Compare Seasonal NDVI Data Across Multiple Years and Visualize Changes

This script compares aggregated seasonal NDVI data across multiple years for a specific
sub-area to identify trends, anomalies, or significant changes in vegetation health.

Usage:
    python scripts/compare_ndvi.py <season> <year1> <year2> ... --sub_area <sub_area_number> [--output OUTPUT_PATH]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from config import SEASONS  # Import SEASONS dictionary

def load_seasonal_ndvi_files(season: str, years: List[int], sub_area: int, data_dir: Path) -> Dict[int, np.ndarray]:
    """
    Load aggregated seasonal NDVI files for specified years and sub-area.

    Parameters
    ----------
    season : str
        Season name.
    years : List[int]
        List of years.
    sub_area : int
        Sub-area number.
    data_dir : Path
        Path to the processed NDVI data directory.

    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping year to NDVI array.
    """
    ndvi_years = {}
    for year in years:
        ndvi_path = data_dir / f"ndvi_{season}_{year}_sub_area_{sub_area}.npy"
        if ndvi_path.exists():
            ndvi = np.load(ndvi_path)
            ndvi_years[year] = ndvi
            logger.debug(f"Loaded NDVI for {season} {year}, Sub-area {sub_area} from {ndvi_path}")
        else:
            logger.warning(f"Aggregated NDVI file not found: {ndvi_path}")
    return ndvi_years

def compute_ndvi_change(ndvi_current: np.ndarray, ndvi_previous: np.ndarray) -> np.ndarray:
    """
    Compute the change in NDVI between two years.

    Parameters
    ----------
    ndvi_current : np.ndarray
        NDVI array for the current year.
    ndvi_previous : np.ndarray
        NDVI array for the previous year.

    Returns
    -------
    np.ndarray
        NDVI change array.
    """
    change = ndvi_current - ndvi_previous
    # Handle NaNs by setting them to zero
    change = np.where(np.isnan(change), 0, change)
    return change

def plot_heatmap(change: np.ndarray, output_path: Path, title: str) -> None:
    """
    Generate and save a heatmap of NDVI changes.

    Parameters
    ----------
    change : np.ndarray
        NDVI change array.
    output_path : Path
        Path to save the heatmap image.
    title : str
        Title of the heatmap.
    """
    plt.figure(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(change, cmap=cmap, center=0, cbar_kws={'label': 'NDVI Change'})
    plt.title(title, fontsize=16)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Heatmap saved to {output_path}")

def main():
    """
    Entry point of the compare_ndvi.py script.
    """
    parser = argparse.ArgumentParser(
        description="Compare aggregated seasonal NDVI data across multiple years for a specific sub-area and visualize changes."
    )
    parser.add_argument(
        "season",
        type=str,
        choices=SEASONS.keys(),
        help="Season to compare (e.g., spring)."
    )
    parser.add_argument(
        "years",
        type=int,
        nargs='+',
        help="List of years to compare (e.g., 2018 2019 2020)."
    )
    parser.add_argument(
        "--sub_area",
        type=int,
        required=True,
        help="Sub-area number."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/ndvi",
        help="Path to the processed NDVI data directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comparisons",
        help="Path to save the heatmap images."
    )

    args = parser.parse_args()

    season = args.season
    years = sorted(args.years)
    sub_area = args.sub_area
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Comparing NDVI for {season.capitalize()} across years {years} in Sub-area {sub_area}.")

    # Load NDVI data for each year
    ndvi_years = load_seasonal_ndvi_files(season, years, sub_area, data_dir)

    # Compare consecutive years
    for i in range(1, len(years)):
        year_prev = years[i - 1]
        year_curr = years[i]
        if year_prev in ndvi_years and year_curr in ndvi_years:
            change = compute_ndvi_change(ndvi_years[year_curr], ndvi_years[year_prev])
            title = f"NDVI Change: {season.capitalize()} {year_prev} to {year_curr} (Sub-area {sub_area})"
            output_path = output_dir / f"ndvi_change_{season}_{year_prev}_to_{year_curr}_sub_area_{sub_area}.png"
            plot_heatmap(change, output_path, title)
        else:
            logger.warning(f"Insufficient NDVI data to compare {year_prev} and {year_curr}.")

if __name__ == "__main__":
    main()
