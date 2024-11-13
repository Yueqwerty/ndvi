# test/test_fetch_ndvi.py
import numpy as np
import pytest
from pathlib import Path
from test.utils import load_npy_file, plot_ndvi

def test_fetch_ndvi():
    """
    Test the NDVI data fetching functionality.
    """
    # Define the path to a sample raw NDVI file
    raw_ndvi_path = Path(r'C:\Users\carlo\Desktop\PYTHON\proyecto_ndvi\data\raw\ndvi\2018-01-01\sub_area_1\ndvi_landsat8.npy')
    
    assert raw_ndvi_path.exists(), f"Raw NDVI file does not exist: {raw_ndvi_path}"
    
    # Load the NDVI data
    ndvi = load_npy_file(raw_ndvi_path)
    
    # Check that the data is not empty
    assert ndvi.size > 0, "Raw NDVI data is empty."
    
    # Check for variability
    unique_values = np.unique(ndvi)
    assert len(unique_values) > 1, "Raw NDVI data has zero variance."
    
    # Optional: Visual inspection
    # plot_ndvi(ndvi, title="Test Raw NDVI Data")
