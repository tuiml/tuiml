"""Reading and writing datasets, in whatever format they arrive.

Every loader returns the same :class:`Dataset` — ``X``, ``y`` and feature
names — so the rest of TuiML never has to care where the data came from.
Each format also has a matching ``save_*``.

Formats
-------
- **CSV / TSV:** :func:`load_csv`, :func:`save_csv`.
- **ARFF:** :func:`load_arff`, :func:`save_arff`. Weka's format, which
  carries column types and nominal values in its header.
- **Parquet:** :func:`load_parquet`, :func:`save_parquet`, plus
  :func:`load_parquet_partitioned` for directory-partitioned datasets.
- **Excel:** :func:`load_excel`, :func:`save_excel`, and
  :func:`load_excel_sheets` for a workbook of several sheets.
- **JSON:** :func:`load_json`, :func:`load_jsonl`, :func:`load_json_nested`
  for records that are not flat, and their ``save_*`` counterparts.
- **NumPy:** :func:`load_numpy`, :func:`save_numpy` (``.npy`` / ``.npz``).
- **pandas:** :func:`from_pandas`, :func:`to_pandas` for in-memory frames.

Detecting the format
--------------------
:func:`load` and :func:`save` pick the right one from the file extension, so
a path is usually all you need. This is what lets ``{"source": "sales.csv"}``
work in a :func:`tuiml.train` spec.

Examples
--------
>>> from tuiml.datasets.loaders import load
>>> data = load("sales.csv", target="label")     # doctest: +SKIP
>>> data.X.shape                                 # doctest: +SKIP
(1000, 12)
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
