# test/test_process_ndvi.py

import pytest
import numpy as np
from pathlib import Path
from test.utils import load_npy_file, load_geotiff

def test_process_ndvi():
    """
    Test the NDVI data processing functionality.
    """
    # Define paths
    raw_ndvi_path = Path(r'C:\Users\carlo\Desktop\PYTHON\proyecto_ndvi\data\raw\ndvi\2018-01-01\sub_area_1\ndvi_landsat8.npy')
    processed_npy_path = Path(r'C:\Users\carlo\Desktop\PYTHON\proyecto_ndvi\data\processed\ndvi\2018-01-01\sub_area_1\ndvi.npy')
    processed_tif_path = Path(r'C:\Users\carlo\Desktop\PYTHON\proyecto_ndvi\data\processed\ndvi\2018-01-01\sub_area_1\ndvi_processed.tif')
    
    # Ensure processed files exist
    assert processed_npy_path.exists(), f"Processed NDVI `.npy` file does not exist: {processed_npy_path}"
    assert processed_tif_path.exists(), f"Processed NDVI GeoTIFF file does not exist: {processed_tif_path}"
    
    # Load and verify processed NDVI data
    ndvi_processed = load_npy_file(processed_npy_path)
    ndvi_processed_tif = load_geotiff(processed_tif_path)
    
    # Check that the processed data is not empty
    assert ndvi_processed.size > 0, "Processed NDVI `.npy` data is empty."
    assert ndvi_processed_tif.size > 0, "Processed NDVI GeoTIFF data is empty."
    
    # Check for variability
    unique_values_npy = np.unique(ndvi_processed)
    assert len(unique_values_npy) > 1, "Processed NDVI `.npy` data has zero variance."
    
    unique_values_tif = np.unique(ndvi_processed_tif)
    assert len(unique_values_tif) > 1, "Processed NDVI GeoTIFF data has zero variance."
    
    # Optional: Visual inspection
    # plot_ndvi(ndvi_processed, title="Test Processed NDVI `.npy` Data")
    # plot_ndvi(ndvi_processed_tif, title="Test Processed NDVI GeoTIFF Data")
