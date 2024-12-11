# scripts/analysis/statistics.py

import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from loguru import logger
import pandas as pd
from scipy import stats

def load_all_statistics(years: List[int], season: str, sub_area: int, data_dir: Path) -> List[Dict]:
    """
    Load all statistical metrics for specified years, season, and sub-area.

    Parameters
    ----------
    years : List[int]
        List of years.
    season : str
        Season name.
    sub_area : int
        Sub-area number.
    data_dir : Path
        Path to the data directory.

    Returns
    -------
    List[Dict]
        List of statistics dictionaries.
    """
    stats_list = []
    for year in years:
        stats_path = data_dir / "statistics" / f"ndvi_statistics_{season}_{year}_sub_area_{sub_area}.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                stats['year'] = year
                stats_list.append(stats)
                logger.debug(f"Loaded statistics from {stats_path}")
        else:
            logger.warning(f"Statistics file not found: {stats_path}")
    return stats_list

def compute_trends(stats_list: List[Dict]) -> Dict[str, float]:
    """
    Compute trends for each statistical metric over the years.

    Parameters
    ----------
    stats_list : List[Dict]
        List of statistics dictionaries sorted by year.

    Returns
    -------
    Dict[str, float]
        Dictionary with trend slopes for each metric.
    """
    if not stats_list:
        logger.error("Statistics list is empty. Cannot compute trends.")
        return {}

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(stats_list)
    df = df.sort_values('year')

    trends = {}
    metrics = ['mean_ndvi', 'median_ndvi', 'std_ndvi', 'min_ndvi', 'max_ndvi']

    for metric in metrics:
        if metric in df.columns:
            slope, intercept, r_value, p_value, std_err = stats.linregress(df['year'], df[metric])
            trends[metric] = slope
            logger.info(f"Trend for {metric}: Slope = {slope:.4f}, R-squared = {r_value**2:.4f}")
        else:
            logger.warning(f"Metric '{metric}' not found in statistics data.")

    return trends
