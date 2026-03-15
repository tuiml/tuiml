"""
Data loaders for various file formats.

Supports: ARFF, CSV, Excel, NumPy, Pandas DataFrame, Parquet, JSON
"""

from tuiml.datasets.loaders.arff import load_arff, save_arff, Dataset
from tuiml.datasets.loaders.csv import load_csv, save_csv
from tuiml.datasets.loaders.numpy import load_numpy, save_numpy
from tuiml.datasets.loaders.pandas import load_pandas, to_pandas, from_pandas
from tuiml.datasets.loaders.excel import load_excel, save_excel, load_excel_sheets
from tuiml.datasets.loaders.parquet import load_parquet, save_parquet, load_parquet_partitioned
from tuiml.datasets.loaders.json import (
    load_json, save_json, load_jsonl, save_jsonl, load_json_nested
)
from tuiml.datasets.loaders.auto import load, save

__all__ = [
    # Dataset container
    "Dataset",
    # ARFF
    "load_arff",
    "save_arff",
    # CSV
    "load_csv",
    "save_csv",
    # NumPy
    "load_numpy",
    "save_numpy",
    # Pandas
    "load_pandas",
    "to_pandas",
    "from_pandas",
    # Excel
    "load_excel",
    "save_excel",
    "load_excel_sheets",
    # Parquet
    "load_parquet",
    "save_parquet",
    "load_parquet_partitioned",
    # JSON
    "load_json",
    "save_json",
    "load_jsonl",
    "save_jsonl",
    "load_json_nested",
    # Auto-detect
    "load",
    "save",
]
