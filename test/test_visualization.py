# test/test_visualization.py

import pytest
import numpy as np
from pathlib import Path
from test.utils import load_geotiff, plot_ndvi

def test_display_ndvi_map():
    """
    Test the NDVI map visualization functionality.
    """
    # Define the path to a processed NDVI GeoTIFF file
    processed_tif_path = Path(r'C:\Users\carlo\Desktop\PYTHON\proyecto_ndvi\data\processed\ndvi\2018-01-01\sub_area_1\ndvi_processed.tif')
    
    assert processed_tif_path.exists(), f"Processed NDVI GeoTIFF file does not exist: {processed_tif_path}"
    
    # Load the NDVI data
    ndvi_processed = load_geotiff(processed_tif_path)
    
    # Check that the data is not empty
    assert ndvi_processed.size > 0, "Processed NDVI GeoTIFF data is empty."
    
    # Check for variability
    unique_values = np.unique(ndvi_processed)
    assert len(unique_values) > 1, "Processed NDVI GeoTIFF data has zero variance."
    
    # Optional: Visual inspection
    # plot_ndvi(ndvi_processed, title="Test NDVI Visualization")
