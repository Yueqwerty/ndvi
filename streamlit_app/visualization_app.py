#!/usr/bin/env python3
"""
Interactive Visualization App for NDVI Data

This script launches a web-based application that allows users to interactively
visualize NDVI data spatially and temporally across different sub-areas and months.

Usage:
    python scripts/visualization_app.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from shapely.geometry import box
from pyproj import Transformer

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

from analysis.statistics import load_all_statistics, compute_trends
from visualization.heatmaps import load_ndvi_geotiff

def create_mapbox_token() -> Optional[str]:
    """
    Retrieve Mapbox access token from environment variables.

    Returns
    -------
    Optional[str]
        Mapbox access token if available, else None.
    """
    return os.getenv("MAPBOX_ACCESS_TOKEN")

def prepare_ndvi_dataframe(ndvi_data: np.ndarray, bounds: List[float]) -> pd.DataFrame:
    """
    Prepare a DataFrame for plotting NDVI data.

    Parameters
    ----------
    ndvi_data : np.ndarray
        NDVI data array.
    bounds : List[float]
        Bounds of the NDVI data [minx, miny, maxx, maxy].

    Returns
    -------
    pd.DataFrame
        DataFrame containing longitude, latitude, and NDVI values.
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

def main():
    """
    Entry point of the visualization_app.py script.
    """
    # Define project directories
    SCRIPT_DIR = Path(__file__).resolve().parent.parent  # scripts/
    PROJECT_ROOT = SCRIPT_DIR.parent  # proyecto_ndvi/
    DATA_DIR = PROJECT_ROOT / 'data'  # proyecto_ndvi/data/
    
    # Load sub-area bounds
    bounds_file = DATA_DIR / 'sub_area_bounds.json'
    if not bounds_file.exists():
        logger.error(f"Archivo de límites de sub-área no encontrado: {bounds_file}")
        sys.exit(1)
    
    with open(bounds_file, 'r') as f:
        sub_area_bounds = json.load(f)
    sub_areas = list(sub_area_bounds.keys())
    
    # Get list of available months
    ndvi_dir = DATA_DIR / "raw" / "ndvi"
    available_months = sorted([d.name for d in ndvi_dir.iterdir() if d.is_dir()])
    
    # Initialize Dash app
    app = dash.Dash(__name__)
    app.title = "NDVI Visualization App"
    
    # Layout of the app
    app.layout = html.Div([
        html.H1("NDVI Data Visualization"),
        
        html.Div([
            html.Div([
                html.Label("Select Months:"),
                dcc.Dropdown(
                    id='months-dropdown',
                    options=[{'label': m, 'value': m} for m in available_months],
                    value=available_months[-3:],  # Seleccionar los últimos 3 meses por defecto
                    multi=True
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label("Select Sub-Area:"),
                dcc.Dropdown(
                    id='sub-area-dropdown',
                    options=[{'label': f"Sub-Area {sa}", 'value': sa} for sa in sub_areas],
                    value=sub_areas[0]
                ),
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        ]),
        
        dcc.Graph(id='ndvi-map'),
        
        html.Div([
            html.H3("NDVI Statistics Trends"),
            dcc.Graph(id='ndvi-trends')
        ], style={'marginTop': '20px'}),
        
        html.Div([
            html.H3("NDVI Difference Heatmap"),
            dcc.Graph(id='ndvi-difference-map')
        ], style={'marginTop': '20px'}),
    ])
    
    @app.callback(
        Output('ndvi-map', 'figure'),
        [Input('months-dropdown', 'value'),
         Input('sub-area-dropdown', 'value')]
    )
    def update_ndvi_map(selected_months: List[str], selected_sub_area: str):
        """
        Actualiza el mapa de NDVI basado en los meses y sub-área seleccionados.
        
        Parameters
        ----------
        selected_months : List[str]
            Lista de meses seleccionados.
        selected_sub_area : str
            Número de sub-área seleccionado.
        
        Returns
        -------
        plotly.graph_objs._figure.Figure
            Figura actualizada del mapa de NDVI.
        """
        if not selected_months:
            return {}
        
        # Para simplificar, tomaremos el promedio de NDVI si se seleccionan múltiples meses
        ndvi_sum = None
        count = 0
        for month in selected_months:
            tif_path = DATA_DIR / "raw" / "ndvi" / month / f"sub_area_{selected_sub_area}" / "ndvi_monthly.tif"
            if not tif_path.exists():
                logger.warning(f"Archivo GeoTIFF no encontrado: {tif_path}")
                continue
            ndvi = load_ndvi_geotiff(tif_path)
            if ndvi_sum is None:
                ndvi_sum = ndvi.copy()
            else:
                ndvi_sum += ndvi
            count += 1
        
        if ndvi_sum is None or count == 0:
            logger.error("No se encontraron archivos GeoTIFF válidos para los meses seleccionados.")
            return {}
        
        ndvi_avg = ndvi_sum / count
        bounds = sub_area_bounds[selected_sub_area]
        df_ndvi = prepare_ndvi_dataframe(ndvi_avg, bounds)
        
        # Crear el mapa de calor
        fig = px.density_mapbox(
            df_ndvi,
            lat='Latitude',
            lon='Longitude',
            z='NDVI',
            radius=10,
            center=dict(lat=df_ndvi['Latitude'].mean(), lon=df_ndvi['Longitude'].mean()),
            zoom=6,
            mapbox_style="open-street-map",
            color_continuous_scale='RdYlGn',
            title=f"Promedio NDVI para {', '.join(selected_months)} - Sub-Area {selected_sub_area}"
        )
        
        mapbox_token = create_mapbox_token()
        if mapbox_token:
            fig.update_layout(mapbox_style="mapbox://styles/mapbox/streets-v11",
                              mapbox_accesstoken=mapbox_token)
        
        return fig
    
    @app.callback(
        Output('ndvi-trends', 'figure'),
        [Input('months-dropdown', 'value'),
         Input('sub-area-dropdown', 'value')]
    )
    def update_ndvi_trends(selected_months: List[str], selected_sub_area: str):
        """
        Actualiza el gráfico de tendencias de estadísticas NDVI.
        
        Parameters
        ----------
        selected_months : List[str]
            Lista de meses seleccionados.
        selected_sub_area : str
            Número de sub-área seleccionado.
        
        Returns
        -------
        plotly.graph_objs._figure.Figure
            Figura actualizada de las tendencias de estadísticas NDVI.
        """
        if not selected_months:
            return {}
        
        trends = compute_trends(load_all_statistics(selected_months, int(selected_sub_area), DATA_DIR))
        metrics = list(trends.keys())
        values = list(trends.values())
        
        fig = go.Figure(data=[
            go.Bar(name='Tendencia', x=metrics, y=values, marker_color='indianred')
        ])
        fig.update_layout(
            title=f"Tendencias de NDVI para {', '.join(selected_months)} - Sub-Area {selected_sub_area}",
            xaxis_title="Métricas NDVI",
            yaxis_title="Pendiente de la Tendencia",
            barmode='group'
        )
        
        return fig
    
    @app.callback(
        Output('ndvi-difference-map', 'figure'),
        [Input('months-dropdown', 'value'),
         Input('sub-area-dropdown', 'value')]
    )
    def update_difference_map(selected_months: List[str], selected_sub_area: str):
        """
        Actualiza el mapa de diferencias de NDVI entre el primer y el último mes seleccionados.
        
        Parameters
        ----------
        selected_months : List[str]
            Lista de meses seleccionados.
        selected_sub_area : str
            Número de sub-área seleccionado.
        
        Returns
        -------
        plotly.graph_objs._figure.Figure
            Figura actualizada del mapa de diferencias de NDVI.
        """
        if len(selected_months) < 2:
            return {}
        
        # Tomar el primer y último mes de la selección
        month1 = selected_months[0]
        month2 = selected_months[-1]
        
        # Rutas a los GeoTIFF
        tif1 = DATA_DIR / "raw" / "ndvi" / month1 / f"sub_area_{selected_sub_area}" / "ndvi_monthly.tif"
        tif2 = DATA_DIR / "raw" / "ndvi" / month2 / f"sub_area_{selected_sub_area}" / "ndvi_monthly.tif"
        
        if not tif1.exists() or not tif2.exists():
            logger.error(f"Uno o ambos archivos GeoTIFF no existen: {tif1}, {tif2}")
            return {}
        
        # Cargar NDVI de ambos meses
        ndvi1 = load_ndvi_geotiff(tif1)
        ndvi2 = load_ndvi_geotiff(tif2)
        
        # Calcular diferencia
        difference = ndvi2 - ndvi1
        bounds = sub_area_bounds[selected_sub_area]
        df_diff = prepare_ndvi_dataframe(difference, bounds)
        
        # Crear el mapa de diferencias
        fig = px.density_mapbox(
            df_diff,
            lat='Latitude',
            lon='Longitude',
            z='NDVI',
            radius=10,
            center=dict(lat=df_diff['Latitude'].mean(), lon=df_diff['Longitude'].mean()),
            zoom=6,
            mapbox_style="open-street-map",
            color_continuous_scale='RdBu',
            title=f"Diferencia NDVI entre {month1} y {month2} - Sub-Area {selected_sub_area}"
        )
        
        mapbox_token = create_mapbox_token()
        if mapbox_token:
            fig.update_layout(mapbox_style="mapbox://styles/mapbox/streets-v11",
                              mapbox_accesstoken=mapbox_token)
        
        return fig
    
    # Run the app
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
