import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

def load_statistics(month: str, sub_area: int, data_dir: Path) -> Dict[str, float]:
    """
    Carga las estadísticas de NDVI para un mes y una sub-área específica.
    
    Parameters
    ----------
    month : str
        Mes en formato 'YYYY-MM'.
    sub_area : int
        Número de sub-área.
    data_dir : Path
        Ruta al directorio de datos.
    
    Returns
    -------
    Dict[str, float]
        Diccionario con estadísticas de NDVI.
    """
    stats_path = data_dir.parent / "results" / "statistics"
    stats_file = stats_path / f"ndvi_statistics_{month}_sub_area_{sub_area}.json"
    
    if not stats_file.exists():
        logger.error(f"Archivo de estadísticas no encontrado: {stats_file}")
        raise FileNotFoundError(f"Archivo de estadísticas no encontrado: {stats_file}")
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    return stats

def load_all_statistics(months: List[str], sub_area: int, data_dir: Path) -> List[Dict[str, float]]:
    """
    Carga las estadísticas de NDVI para múltiples meses y una sub-área específica.
    
    Parameters
    ----------
    months : List[str]
        Lista de meses en formato 'YYYY-MM'.
    sub_area : int
        Número de sub-área.
    data_dir : Path
        Ruta al directorio de datos.
    
    Returns
    -------
    List[Dict[str, float]]
        Lista de diccionarios con estadísticas de NDVI por mes.
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
    Calcula las tendencias de NDVI a partir de una lista de estadísticas.
    
    Parameters
    ----------
    stats_list : List[Dict[str, float]]
        Lista de diccionarios con estadísticas de NDVI por mes.
    
    Returns
    -------
    Dict[str, float]
        Diccionario con tendencias calculadas.
    """
    if not stats_list:
        logger.error("La lista de estadísticas está vacía.")
        raise ValueError("La lista de estadísticas está vacía.")
    
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
