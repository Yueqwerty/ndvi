# test/utils.py

import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt

def load_npy_file(file_path: Path) -> np.ndarray:
    """
    Load NDVI data from a `.npy` file.
    
    :param file_path: Path to the `.npy` file.
    :return: NDVI data as a NumPy array.
    """
    return np.load(file_path)

def load_geotiff(file_path: Path) -> np.ndarray:
    """
    Load NDVI data from a GeoTIFF file.
    
    :param file_path: Path to the GeoTIFF file.
    :return: NDVI data as a NumPy array.
    """
    with rasterio.open(file_path) as src:
        ndvi = src.read(1)
        ndvi = np.where(ndvi == src.nodata, np.nan, ndvi)
    return ndvi

def plot_ndvi(ndvi: np.ndarray, title: str = "NDVI Visualization"):
    """
    Plot NDVI data using Matplotlib.
    
    :param ndvi: NDVI data as a NumPy array.
    :param title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(ndvi, cmap='RdYlGn')
    plt.colorbar(label='NDVI')
    plt.title(title)
    plt.xlabel('Pixel Column')
    plt.ylabel('Pixel Row')
    plt.show()
