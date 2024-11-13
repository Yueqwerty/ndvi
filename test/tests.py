import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Define the path to the raw NDVI file
raw_ndvi_path = Path(r'C:\Users\carlo\Desktop\PYTHON\proyecto_ndvi\data\raw\ndvi\2018-06-16\sub_area_1\ndvi_landsat8.npy')

# Check if the file exists
if raw_ndvi_path.exists():
    try:
        ndvi_raw = np.load(raw_ndvi_path)
        print("Raw NDVI Data Loaded Successfully.")
        print(f"Shape: {ndvi_raw.shape}")
        print(f"Min: {np.nanmin(ndvi_raw)}")
        print(f"Max: {np.nanmax(ndvi_raw)}")
        print(f"Mean: {np.nanmean(ndvi_raw)}")
        print(f"Std Dev: {np.nanstd(ndvi_raw)}")

        # Visualize the NDVI data
        plt.imshow(ndvi_raw, cmap='RdYlGn')
        plt.colorbar(label='NDVI')
        plt.title('Raw NDVI Data')
        plt.show()
    except Exception as e:
        print(f"Error loading NDVI data: {e}")
else:
    print(f"Raw NDVI file does not exist: {raw_ndvi_path}")
