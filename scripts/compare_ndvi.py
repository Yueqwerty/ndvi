#!/usr/bin/env python3
"""
Compare NDVI Statistics or Spatial Data Between Two Periods

This script compares NDVI data (statistics or raw) between two periods.
It can generate:
- Statistical comparisons (differences in mean, median, etc.).
- Pixel-wise differences as GeoTIFF.
- Heatmaps for the entire area to visualize changes.
- Automatic visualization of the comparison results.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_statistics(period_dir: Path) -> Dict[int, Dict]:
    """
    Load statistics JSON files for a given period with error handling.
    """
    stats = {}
    try:
        stat_files = list(period_dir.glob("ndvi_statistics_*.json"))
        if not stat_files:
            logger.warning(f"No statistics files found in {period_dir}")
            return stats

        for file_path in stat_files:
            try:
                sub_area = int(file_path.stem.split("_")[-1])
                with open(file_path, "r") as f:
                    stats[sub_area] = json.load(f)
            except (ValueError, json.JSONDecodeError) as e:
                logger.error(f"Error loading statistics file {file_path}: {e}")
        
        return stats
    except Exception as e:
        logger.error(f"Unexpected error loading statistics: {e}")
        return {}

def load_ndvi_array(file_path: Path) -> Optional[np.ndarray]:
    """
    Load NDVI array from a .npy file with robust error handling.
    """
    try:
        return np.load(file_path)
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Error loading NDVI array from {file_path}: {e}")
        return None

def compare_ndvi_arrays(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    Compare NDVI arrays (pixel-wise difference).

    Parameters
    ----------
    array1 : np.ndarray
        First NDVI array.
    array2 : np.ndarray
        Second NDVI array.

    Returns
    -------
    np.ndarray
        Difference array (array2 - array1).
    """
    return array2 - array1

def generate_comparison_heatmap(differences: np.ndarray, output_path: Path):
    """
    Generate a heatmap for NDVI differences across all sub-areas.

    Parameters
    ----------
    differences : np.ndarray
        NDVI difference array.
    output_path : Path
        Path to save the heatmap image.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        differences, 
        cmap="coolwarm", 
        cbar_kws={'label': 'NDVI Difference'}, 
        center=0,
        annot=False,
        fmt=".2f"
    )
    plt.title("NDVI Differences Heatmap", fontsize=16)
    plt.xlabel("Columns", fontsize=12)
    plt.ylabel("Rows", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved heatmap: {output_path}")

def save_difference_as_geotiff(
    diff_array: np.ndarray, 
    output_path: Path, 
    ref_raster_path: Path
):
    """
    Save NDVI difference array as a GeoTIFF with comprehensive error handling.
    """
    try:
        with rasterio.open(ref_raster_path) as src:
            profile = src.profile
            profile.update(dtype=diff_array.dtype, count=1)
            
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(diff_array, 1)
        logger.info(f"Successfully saved GeoTIFF: {output_path}")
    except Exception as e:
        logger.error(f"Error saving GeoTIFF {output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare NDVI data between two periods."
    )
    parser.add_argument("period1", type=str, help="First period (e.g., '2019-03').")
    parser.add_argument("period2", type=str, help="Second period (e.g., '2020-03').")
    args = parser.parse_args()

    data_dir = Path("data/processed/ndvi")
    period1_dir = data_dir / args.period1
    period2_dir = data_dir / args.period2
    output_dir = Path("results/comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_difference = None

    try:
        for sub_area_dir in period1_dir.glob("ndvi_statistics_*.json"):
            try:
                sub_area = int(sub_area_dir.stem.split("_")[-1])
                array1_path = period1_dir / f"sub_area_{sub_area}/ndvi_monthly.npy"
                array2_path = period2_dir / f"sub_area_{sub_area}/ndvi_monthly.npy"

                array1 = load_ndvi_array(array1_path)
                array2 = load_ndvi_array(array2_path)

                if array1 is None or array2 is None:
                    logger.warning(f"Skipping sub-area {sub_area} due to missing data")
                    continue

                diff_array = compare_ndvi_arrays(array1, array2)

                # Combine differences into one large array for heatmap
                if combined_difference is None:
                    combined_difference = diff_array
                else:
                    combined_difference = np.vstack((combined_difference, diff_array))

                # Save GeoTIFF per sub-area
                ref_raster = array1_path.with_suffix(".tif")
                diff_output_path = output_dir / f"ndvi_difference_{args.period1}_vs_{args.period2}_sub_area_{sub_area}.tif"
                save_difference_as_geotiff(diff_array, diff_output_path, ref_raster)
            
            except Exception as sub_area_error:
                logger.error(f"Error processing sub-area {sub_area}: {sub_area_error}")

        # Generate overall heatmap
        if combined_difference is not None:
            heatmap_path = output_dir / f"ndvi_difference_heatmap_{args.period1}_vs_{args.period2}.png"
            generate_comparison_heatmap(combined_difference, heatmap_path)

    except Exception as e:
        logger.error(f"Unexpected error in main execution: {e}")

if __name__ == "__main__":
    main()
