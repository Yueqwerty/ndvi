# scripts/analysis/statistics.py

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger

def load_statistics(month: str, sub_area: int, data_dir: Path) -> Dict[str, float]:
    """
    Load NDVI statistics for a specific month and sub-area.

    :param month: Month in 'YYYY-MM' format.
    :type month: str
    :param sub_area: Sub-area number.
    :type sub_area: int
    :param data_dir: Path to the data directory.
    :type data_dir: Path
    :return: Dictionary containing NDVI statistics.
    :rtype: Dict[str, float]
    :raises FileNotFoundError: If the statistics file does not exist.
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

    :param months: List of months in 'YYYY-MM' format.
    :type months: List[str]
    :param sub_area: Sub-area number.
    :type sub_area: int
    :param data_dir: Path to the data directory.
    :type data_dir: Path
    :return: List of dictionaries containing NDVI statistics per month.
    :rtype: List[Dict[str, float]]
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

    :param stats_list: List of dictionaries containing NDVI statistics per month.
    :type stats_list: List[Dict[str, float]]
    :return: Dictionary containing computed trends.
    :rtype: Dict[str, float]
    :raises ValueError: If the statistics list is empty.
    """
    if not stats_list:
        logger.error("Statistics list is empty.")
        raise ValueError("Statistics list is empty.")
    
    df = np.array([[
        stat['mean_ndvi'], 
        stat['median_ndvi'], 
        stat['std_ndvi'], 
        stat['min_ndvi'], 
        stat['max_ndvi'], 
        stat['count_valid_pixels']
    ] for stat in stats_list])
    
    trends = {
        "trend_mean_ndvi": np.polyfit(range(len(df)), df[:,0], 1)[0],
        "trend_median_ndvi": np.polyfit(range(len(df)), df[:,1], 1)[0],
        "trend_std_ndvi": np.polyfit(range(len(df)), df[:,2], 1)[0],
        "trend_min_ndvi": np.polyfit(range(len(df)), df[:,3], 1)[0],
        "trend_max_ndvi": np.polyfit(range(len(df)), df[:,4], 1)[0],
        "trend_count_valid_pixels": np.polyfit(range(len(df)), df[:,5], 1)[0],
    }
    return trends
