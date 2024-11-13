import numpy as np
from typing import List, Dict

def calculate_trend(values: List[float]) -> float:
    """
    Calcula la pendiente de la línea de tendencia utilizando regresión lineal.
    
    Parameters
    ----------
    values : List[float]
        Lista de valores numéricos.
    
    Returns
    -------
    float
        Pendiente de la línea de tendencia.
    """
    x = np.arange(len(values))
    y = np.array(values)
    slope, _ = np.polyfit(x, y, 1)
    return slope
