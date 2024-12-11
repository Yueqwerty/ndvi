import numpy as np
from typing import List

def calculate_trend(values: List[float]) -> float:
    """
    Calculate the trend (slope) of a given list of values using linear regression.

    Parameters
    ----------
    values : List[float]
        List of numerical values.

    Returns
    -------
    float
        The slope of the trend line.
    """
    x = np.arange(len(values))
    y = np.array(values)
    slope, _ = np.polyfit(x, y, 1)
    return slope
