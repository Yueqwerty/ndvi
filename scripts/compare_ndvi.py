#!/usr/bin/env python3
"""
Compare NDVI Data Across Multiple Months

This script loads computed NDVI statistics for a given sub-area across multiple months
and calculates trends. It can help identify long-term vegetation changes, such as
forestation (increasing NDVI) or deforestation (decreasing NDVI).

Usage:
    python scripts/compare_ndvi.py YYYY-MM1 YYYY-MM2 ... --sub_area SUB_AREA_NUMBER [--output OUTPUT_PATH]

Example:
    python scripts/compare_ndvi.py 2018-03 2019-03 2020-03 --sub_area 1
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from analysis.statistics import load_all_statistics, compute_trends

def compare_multiple_months(months: List[str], sub_area: int, data_dir: Path) -> Dict[str, float]:
    """
    Compare multiple months and calculate NDVI trends.

    Parameters
    ----------
    months : List[str]
        List of months in 'YYYY-MM' format.
    sub_area : int
        Sub-area number.
    data_dir : Path
        Path to the data directory.

    Returns
    -------
    Dict[str, float]
        Dictionary of computed trends.
    """
    # Load statistics for each month and sub-area
    stats_list = load_all_statistics(months, sub_area, data_dir)

    if not stats_list:
        logger.error("No statistics found for the given months and sub-area. Cannot compute trends.")
        sys.exit(1)

    # Compute trends over time (e.g., slope for mean_ndvi, median_ndvi, etc.)
    trends = compute_trends(stats_list)
    return trends

def plot_trends(trends: Dict[str, float], output_path: Path) -> None:
    """
    Plot NDVI trends and save as PNG.

    Parameters
    ----------
    trends : Dict[str, float]
        Computed trends dictionary.
    output_path : Path
        Path to save the plot.
    """
    metrics = list(trends.keys())
    values = list(trends.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics, values, color='seagreen')
    plt.xlabel('NDVI Metrics')
    plt.ylabel('Trend (Slope)')
    plt.title('NDVI Trends Over Time')

    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Trend plot saved at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare aggregated NDVI data across multiple months.")
    parser.add_argument("months", type=str, nargs='+', help="List of months in YYYY-MM format.")
    parser.add_argument("--sub_area", type=int, required=True, help="Sub-area number.")
    parser.add_argument("--output", type=str, default=None, help="Output path for the trends plot.")

    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    DATA_DIR = PROJECT_ROOT / 'data'  # Data directory containing 'raw', 'processed'
    RESULTS_DIR = PROJECT_ROOT / 'results' / 'comparisons'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Compare NDVI data across the specified months
    trends = compare_multiple_months(args.months, args.sub_area, DATA_DIR)

    # Determine output path for the trend plot
    if args.output:
        output_path = Path(args.output)
    else:
        months_str = '_vs_'.join(args.months)
        output_filename = f"ndvi_trends_{months_str}_sub_area_{args.sub_area}.png"
        output_path = RESULTS_DIR / output_filename

    # Plot and save the trends
    plot_trends(trends, output_path)

if __name__ == "__main__":
    main()
