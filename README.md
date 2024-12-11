# Proyecto NDVI

**Proyecto NDVI** is a project aimed at analyzing and visualizing changes in the Normalized Difference Vegetation Index (NDVI) over time across various sub-areas in the Aysén region, specifically in Coyhaique. The project leverages satellite imagery data to monitor vegetation health and trends, accounting for the unique seasonal patterns of the Southern Hemisphere.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Fetch and Process NDVI Data](#fetch-and-process-ndvi-data)
  - [Compare Seasonal NDVI Changes](#compare-seasonal-ndvi-changes)
  - [Interactive Visualization](#interactive-visualization)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Fetching:** Retrieve NDVI data from Sentinel Hub API.
- **Data Processing:** Aggregate and compute statistics for NDVI data.
- **Seasonal Aggregation:** Select and aggregate the best images per season and year.
- **Trend Analysis:** Calculate trends over time for various NDVI metrics.
- **Visualization:** Generate heatmaps to visualize changes in NDVI.
- **Interactive Dashboard:** Explore NDVI data through a Streamlit application.

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/proyecto_ndvi.git
    cd proyecto_ndvi
    ```

2. **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Environment Variables:**

    Create a `.env` file in the root directory with the following content:

    ```env
    # Sentinel Hub Credentials
    SENTINELHUB_CLIENT_ID=your_client_id
    SENTINELHUB_CLIENT_SECRET=your_client_secret
    SENTINELHUB_TOKEN_URL=https://services.sentinel-hub.com/oauth/token
    SENTINELHUB_PROCESS_URL=https://services-uswest2.sentinel-hub.com/api/v1/process

    # Optional: Log Level
    FETCH_NDVI_LOG_LEVEL=DEBUG
    ```

## Usage

### Fetch and Process NDVI Data

Fetch NDVI data for a specific month and process it to compute statistical metrics. Optionally, perform seasonal aggregation.

```bash
python scripts/process_ndvi.py <YYYY-MM> --sub_area <SUB_AREA_NUMBER> [--season <SEASON>] [--aggregate_season] [--method <mean|median|max>]
