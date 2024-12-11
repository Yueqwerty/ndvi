#!/usr/bin/env python3
"""
Interactive Visualization App for NDVI Data

This script launches a Streamlit-based web application that allows users to interactively
visualize NDVI data spatially and temporally across different sub-areas and months.

Usage:
    streamlit run streamlit_app/visualization_app.py
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import os
import json
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from shapely.geometry import box
from pyproj import Transformer

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from scripts.analysis.statistics import load_all_statistics, compute_trends
from scripts.visualization.heatmaps import load_ndvi_geotiff, prepare_ndvi_dataframe

def create_mapbox_token() -> Optional[str]:
    """
    Retrieve Mapbox access token from environment variables.

    :return: Mapbox access token if available, else None.
    :rtype: Optional[str]
    """
    return os.getenv("MAPBOX_ACCESS_TOKEN")

def main():
    """
    Entry point of the visualization_app.py script.
    """
    # Define project directories
    SCRIPT_DIR = Path(__file__).resolve().parent.parent  # streamlit_app/
    PROJECT_ROOT = SCRIPT_DIR.parent  # proyecto_ndvi/
    DATA_DIR = PROJECT_ROOT / 'data'  # proyecto_ndvi/data/

    # Set up logging
    logger.add(PROJECT_ROOT / 'logs' / 'visualize_ndvi.log', rotation="10 MB")

    # Load sub-area bounds
    bounds_file = DATA_DIR / 'sub_area_bounds.json'
    if not bounds_file.exists():
        logger.error(f"Sub-area bounds file not found: {bounds_file}")
        st.error("Sub-area bounds file not found. Please ensure 'sub_area_bounds.json' exists in the data directory.")
        sys.exit(1)

    with open(bounds_file, 'r') as f:
        sub_area_bounds = json.load(f)
    sub_areas = sorted(sub_area_bounds.keys(), key=lambda x: int(x))

    # Get list of available months
    ndvi_dir = DATA_DIR / "raw" / "ndvi"
    if not ndvi_dir.exists():
        logger.error(f"NDVI raw data directory not found: {ndvi_dir}")
        st.error("NDVI raw data directory not found. Please ensure 'data/raw/ndvi/' exists.")
        sys.exit(1)

    available_months = sorted([d.name for d in ndvi_dir.iterdir() if d.is_dir()])

    # Streamlit App Layout
    st.title("NDVI Data Visualization")

    st.sidebar.header("Configuration")

    # Select Months
    selected_months = st.sidebar.multiselect(
        "Select Months:",
        options=available_months,
        default=available_months[-3:] if len(available_months) >= 3 else available_months
    )

    # Select Sub-Area
    selected_sub_area = st.sidebar.selectbox(
        "Select Sub-Area:",
        options=[f"Sub-Area {sa}" for sa in sub_areas],
        index=0
    )
    # Extract sub-area number as string
    sub_area_num = selected_sub_area.split(" ")[-1]

    st.sidebar.markdown("---")

    # Display selected configurations
    st.sidebar.markdown(f"**Selected Sub-Area:** {selected_sub_area}")
    st.sidebar.markdown(f"**Selected Months:** {', '.join(selected_months)}")

    # Average NDVI Map
    st.header("Average NDVI Map")

    if selected_months:
        ndvi_sum = None
        count = 0
        for month in selected_months:
            tif_path = ndvi_dir / month / f"sub_area_{sub_area_num}" / "ndvi_monthly.tif"
            if not tif_path.exists():
                logger.warning(f"GeoTIFF file not found: {tif_path}")
                st.warning(f"GeoTIFF file not found for {month}. Skipping.")
                continue
            ndvi = load_ndvi_geotiff(tif_path)
            if ndvi is None:
                st.warning(f"Failed to load NDVI data for {month}. Skipping.")
                continue
            if ndvi_sum is None:
                ndvi_sum = ndvi.copy()
            else:
                ndvi_sum += ndvi
            count += 1

        if ndvi_sum is not None and count > 0:
            ndvi_avg = ndvi_sum / count
            bounds = sub_area_bounds[sub_area_num]
            df_ndvi = prepare_ndvi_dataframe(ndvi_avg, bounds)

            # Create the NDVI average map
            fig_ndvi = px.density_mapbox(
                df_ndvi,
                lat='Latitude',
                lon='Longitude',
                z='NDVI',
                radius=10,
                center=dict(lat=df_ndvi['Latitude'].mean(), lon=df_ndvi['Longitude'].mean()),
                zoom=6,
                mapbox_style="open-street-map",
                color_continuous_scale='RdYlGn',
                title=f"Average NDVI for {', '.join(selected_months)} - {selected_sub_area}"
            )

            mapbox_token = create_mapbox_token()
            if mapbox_token:
                fig_ndvi.update_layout(mapbox_style="mapbox://styles/mapbox/streets-v11",
                                      mapbox_accesstoken=mapbox_token)

            st.plotly_chart(fig_ndvi, use_container_width=True)
        else:
            st.warning("No valid NDVI data available for the selected months and sub-area.")
    else:
        st.warning("Please select at least one month to display the NDVI map.")

    # NDVI Statistics Trends
    st.header("NDVI Statistics Trends")

    if selected_months:
        try:
            stats_list = load_all_statistics(selected_months, int(sub_area_num), DATA_DIR)
            trends = compute_trends(stats_list)

            if trends:
                metrics = list(trends.keys())
                values = list(trends.values())

                fig_trends = go.Figure(data=[
                    go.Bar(name='Trend', x=metrics, y=values, marker_color='indianred')
                ])
                fig_trends.update_layout(
                    title=f"NDVI Trends for {', '.join(selected_months)} - {selected_sub_area}",
                    xaxis_title="NDVI Metrics",
                    yaxis_title="Trend Slope",
                    barmode='group'
                )

                st.plotly_chart(fig_trends, use_container_width=True)
            else:
                st.warning("No trend data available.")
        except Exception as e:
            logger.error(f"Error computing trends: {e}")
            st.error("An error occurred while computing trends.")
    else:
        st.warning("Please select at least one month to display trends.")

    # NDVI Difference Heatmap
    st.header("NDVI Difference Heatmap")

    if len(selected_months) >= 2:
        # Take the first and last selected month
        month1 = selected_months[0]
        month2 = selected_months[-1]

        tif1 = ndvi_dir / month1 / f"sub_area_{sub_area_num}" / "ndvi_monthly.tif"
        tif2 = ndvi_dir / month2 / f"sub_area_{sub_area_num}" / "ndvi_monthly.tif"

        if not tif1.exists() or not tif2.exists():
            logger.error(f"One or both GeoTIFF files do not exist: {tif1}, {tif2}")
            st.error("One or both GeoTIFF files for the selected periods are missing.")
        else:
            ndvi1 = load_ndvi_geotiff(tif1)
            ndvi2 = load_ndvi_geotiff(tif2)

            if ndvi1 is None or ndvi2 is None:
                st.error("Failed to load NDVI data for one or both selected periods.")
            else:
                # Calculate difference
                difference = ndvi2 - ndvi1
                bounds = sub_area_bounds[sub_area_num]
                df_diff = prepare_ndvi_dataframe(difference, bounds)

                # Create the difference heatmap
                fig_diff = px.density_mapbox(
                    df_diff,
                    lat='Latitude',
                    lon='Longitude',
                    z='NDVI',
                    radius=10,
                    center=dict(lat=df_diff['Latitude'].mean(), lon=df_diff['Longitude'].mean()),
                    zoom=6,
                    mapbox_style="open-street-map",
                    color_continuous_scale='RdBu',
                    title=f"NDVI Difference between {month2} and {month1} - {selected_sub_area}"
                )

                mapbox_token = create_mapbox_token()
                if mapbox_token:
                    fig_diff.update_layout(mapbox_style="mapbox://styles/mapbox/streets-v11",
                                          mapbox_accesstoken=mapbox_token)

                st.plotly_chart(fig_diff, use_container_width=True)
    else:
        st.warning("Please select at least two months to display the NDVI difference heatmap.")

if __name__ == "__main__":
    main()
