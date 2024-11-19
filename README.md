# Proyecto NDVI

**Proyecto NDVI** is a project aimed at analyzing and visualizing changes in the Normalized Difference Vegetation Index (NDVI) over time across various sub-areas. The project leverages satellite imagery data to monitor vegetation health and trends.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Fetching:** Retrieve NDVI data from Sentinel Hub API.
- **Data Processing:** Aggregate and compute statistics for NDVI data.
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

### Fetch NDVI Data

Run the script to fetch NDVI data for a specific month:

```bash
python scripts/fetch_ndvi.py YYYY-MM
