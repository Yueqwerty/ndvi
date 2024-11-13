#!/usr/bin/env python3
"""
Compare NDVI Data Across Multiple Months or Sub-Areas

This script compares aggregated monthly NDVI data across multiple months or
different sub-areas to identify trends, anomalies, or significant changes in
vegetation health.

Usage:
    python scripts/compare_ndvi.py YYYY-MM1 YYYY-MM2 ... [--sub_area SUB_AREA_NUMBER] [--method METHOD] [--output OUTPUT_PATH]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from analysis.statistics import load_all_statistics, compute_trends

def compare_multiple_months(months: List[str], sub_area: int, data_dir: Path, method: str) -> Dict[str, float]:
    """
    Compara múltiples meses y calcula tendencias de NDVI.

    Parameters
    ----------
    months : List[str]
        Lista de meses en formato 'YYYY-MM'.
    sub_area : int
        Número de sub-área.
    data_dir : Path
        Ruta al directorio de datos.
    method : str
        Método de comparación.

    Returns
    -------
    Dict[str, float]
        Diccionario con tendencias calculadas.
    """
    stats_list = load_all_statistics(months, sub_area, data_dir)
    trends = compute_trends(stats_list)
    return trends

def plot_trends(trends: Dict[str, float], method: str, output_path: Path) -> None:
    """
    Plotea las tendencias de NDVI.

    Parameters
    ----------
    trends : Dict[str, float]
        Diccionario con tendencias calculadas.
    method : str
        Método de comparación.
    output_path : Path
        Ruta para guardar el gráfico.
    """
    metrics = list(trends.keys())
    values = list(trends.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics, values, color='seagreen')
    plt.xlabel('Métricas NDVI')
    plt.ylabel('Tendencia')
    plt.title(f'Tendencias de NDVI ({method.capitalize()})')
    
    # Anotar barras con valores
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # Desplazamiento vertical de 3 puntos
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Gráfico de tendencias guardado en {output_path}")

def main():
    """
    Entry point of the compare_ndvi.py script.
    """
    parser = argparse.ArgumentParser(
        description="Compare aggregated NDVI data across multiple months or sub-areas."
    )
    parser.add_argument(
        "months",
        type=str,
        nargs='+',
        help="Lista de meses para comparación en formato YYYY-MM."
    )
    parser.add_argument(
        "--sub_area",
        type=int,
        default=None,
        help="Número de sub-área para comparar. Si no se proporciona, se compararán todas las sub-áreas individualmente."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["trend"],
        default="trend",
        help="Método de comparación a usar. Actualmente solo 'trend' está soportado."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta para guardar el gráfico. Si no se proporciona, se guardará en 'results/comparisons/'."
    )

    args = parser.parse_args()

    # Definir directorios del proyecto
    SCRIPT_DIR = Path(__file__).resolve().parent  # scripts/
    PROJECT_ROOT = SCRIPT_DIR.parent  # proyecto_ndvi/
    DATA_DIR = PROJECT_ROOT / 'data'  # proyecto_ndvi/data/
    RESULTS_DIR = PROJECT_ROOT / 'results' / 'comparisons'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.sub_area:
        logger.error("Se debe especificar una sub-área con --sub_area para comparación múltiple de meses.")
        sys.exit(1)

    # Comparar múltiples meses
    trends = compare_multiple_months(args.months, args.sub_area, DATA_DIR, args.method)

    # Definir ruta de salida
    if args.output:
        output_path = Path(args.output)
    else:
        months_str = '_vs_'.join(args.months)
        output_filename = f"ndvi_trends_{months_str}_sub_area_{args.sub_area}.png"
        output_path = RESULTS_DIR / output_filename

    # Plotear tendencias
    plot_trends(trends, args.method, output_path)

if __name__ == "__main__":
    main()
