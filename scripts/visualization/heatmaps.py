# scripts/visualization/heatmaps.py

import rasterio
import numpy as np
from pathlib import Path
from typing import Optional, List
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

def load_ndvi_geotiff(tif_path: Path) -> Optional[np.ndarray]:
    """
    Load NDVI data from a GeoTIFF file.

    :param tif_path: Path to the GeoTIFF file.
    :type tif_path: Path
    :return: NDVI data as a NumPy array.
    :rtype: Optional[np.ndarray]
    """
    try:
        with rasterio.open(tif_path) as src:
            ndvi = src.read(1).astype(float)
            ndvi[ndvi == src.nodata] = np.nan
        return ndvi
    except Exception as e:
        logger.error(f"Error loading GeoTIFF file {tif_path}: {e}")
        return None

def prepare_ndvi_dataframe(ndvi_data: np.ndarray, bounds: List[float]) -> pd.DataFrame:
    """
    Prepare a DataFrame for plotting NDVI data.

    :param ndvi_data: NDVI data array.
    :type ndvi_data: np.ndarray
    :param bounds: Bounds of the NDVI data [minx, miny, maxx, maxy].
    :type bounds: List[float]
    :return: DataFrame containing longitude, latitude, and NDVI values.
    :rtype: pd.DataFrame
    """
    minx, miny, maxx, maxy = bounds
    height, width = ndvi_data.shape
    lon = np.linspace(minx, maxx, width)
    lat = np.linspace(miny, maxy, height)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    df = pd.DataFrame({
        'Longitude': lon_grid.flatten(),
        'Latitude': lat_grid.flatten(),
        'NDVI': ndvi_data.flatten()
    })
    df = df.dropna(subset=['NDVI'])
    return df

def plot_ndvi_heatmap(ndvi_data: np.ndarray, bounds: List[float], output_path: Path, title: str) -> None:
    """
    Plot and save a heatmap of NDVI data.

    :param ndvi_data: NDVI data array.
    :type ndvi_data: np.ndarray
    :param bounds: Bounds of the NDVI data [minx, miny, maxx, maxy].
    :type bounds: List[float]
    :param output_path: Path to save the heatmap image.
    :type output_path: Path
    :param title: Title of the heatmap.
    :type title: str
    """
    plt.figure(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(ndvi_data, cmap=cmap, center=0, cbar_kws={'label': 'NDVI'})
    plt.title(title, fontsize=16)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"NDVI heatmap saved to {output_path}")

def plot_ndvi_change_heatmap(change_data: np.ndarray, output_path: Path, title: str) -> None:
    """
    Plot and save a heatmap of NDVI changes.

    :param change_data: NDVI change array.
    :type change_data: np.ndarray
    :param output_path: Path to save the heatmap image.
    :type output_path: Path
    :param title: Title of the heatmap.
    :type title: str
    """
    plt.figure(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(change_data, cmap=cmap, center=0, cbar_kws={'label': 'NDVI Change'})
    plt.title(title, fontsize=16)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"NDVI change heatmap saved to {output_path}")
