import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger

def load_statistics(month: str, sub_area: int, data_dir: Path) -> Dict[str, float]:
    """
    Load NDVI statistics for a specific month and sub-area.

    Parameters
    ----------
    month : str
        Month in 'YYYY-MM' format.
    sub_area : int
        Sub-area number.
    data_dir : Path
        Path to the data directory.

    Returns
    -------
    Dict[str, float]
        Dictionary containing NDVI statistics.

    Raises
    ------
    FileNotFoundError
        If the statistics file does not exist.
    """
    stats_path = data_dir.parent / "results" / "statistics"
    stats_file = stats_path / f"ndvi_statistics_{month}_sub_area_{sub_area}.json"
    
    if not stats_file.exists():
        logger.error(f"Statistics file not found: {stats_file}")
        raise FileNotFoundError(f"Statistics file not found: {stats_file}")
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    return stats

def load_all_statistics(months: List[str], sub_area: int, data_dir: Path) -> List[Dict[str, float]]:
    """
    Load NDVI statistics for multiple months and a specific sub-area.

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
    List[Dict[str, float]]
        List of dictionaries containing NDVI statistics per month.
    """
    stats_list = []
    for month in months:
        try:
            stats = load_statistics(month, sub_area, data_dir)
            stats['month'] = month
            stats_list.append(stats)
        except FileNotFoundError as e:
            logger.warning(e)
            continue
    return stats_list

def compute_trends(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute NDVI trends from a list of statistics.

    Parameters
    ----------
    stats_list : List[Dict[str, float]]
        List of statistics dictionaries per month.

    Returns
    -------
    Dict[str, float]
        Dictionary containing computed trends for various NDVI metrics.

    Raises
    ------
    ValueError
        If the statistics list is empty.
    """
    if not stats_list:
        logger.error("Statistics list is empty.")
        raise ValueError("Statistics list is empty.")
    
    # Ensure that all stats have the required keys
    required_keys = ["mean_ndvi", "median_ndvi", "std_dev_ndvi", "min_ndvi", "max_ndvi", "count_valid_pixels"]
    for stat in stats_list:
        for rk in required_keys:
            if rk not in stat:
                logger.error(f"Key {rk} not found in statistics.")
                raise ValueError("Invalid statistics structure.")

    df = np.array([[
        stat['mean_ndvi'], 
        stat['median_ndvi'], 
        stat['std_dev_ndvi'], 
        stat['min_ndvi'], 
        stat['max_ndvi'], 
        stat['count_valid_pixels']
    ] for stat in stats_list])
    
    # Compute linear trend (slope) for each metric
    x = np.arange(len(df))
    trends = {}
    metrics = ["mean_ndvi", "median_ndvi", "std_dev_ndvi", "min_ndvi", "max_ndvi", "count_valid_pixels"]
    for i, m in enumerate(metrics):
        slope, _ = np.polyfit(x, df[:, i], 1)
        trends[f"trend_{m}"] = slope

    return trends
