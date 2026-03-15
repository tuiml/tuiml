"""
Auto-detect file format and load/save accordingly.
"""

import numpy as np
from typing import List, Optional, Union
from pathlib import Path

from tuiml.datasets.loaders.arff import Dataset, load_arff, save_arff
from tuiml.datasets.loaders.csv import load_csv, save_csv
from tuiml.datasets.loaders.numpy import load_numpy, save_numpy
from tuiml.datasets.loaders.excel import load_excel, save_excel
from tuiml.datasets.loaders.parquet import load_parquet, save_parquet
from tuiml.datasets.loaders.json import load_json, save_json, load_jsonl

# File extension to loader mapping
LOADERS = {
    '.arff': load_arff,
    '.csv': load_csv,
    '.tsv': lambda f, **kw: load_csv(f, delimiter='\t', **kw),
    '.npy': load_numpy,
    '.npz': load_numpy,
    '.xlsx': load_excel,
    '.xls': load_excel,
    '.parquet': load_parquet,
    '.pq': load_parquet,
    '.json': load_json,
    '.jsonl': load_jsonl,
    '.ndjson': load_jsonl,
}

# File extension to saver mapping
SAVERS = {
    '.arff': save_arff,
    '.csv': save_csv,
    '.tsv': lambda f, **kw: save_csv(f, delimiter='\t', **kw),
    '.npy': save_numpy,
    '.npz': save_numpy,
    '.xlsx': save_excel,
    '.xls': save_excel,
    '.parquet': save_parquet,
    '.pq': save_parquet,
    '.json': save_json,
    '.jsonl': lambda f, **kw: save_json(f, lines=True, **kw),
    '.ndjson': lambda f, **kw: save_json(f, lines=True, **kw),
}

def load(
    filepath: Union[str, Path],
    **kwargs
) -> Dataset:
    """Load data from a file with auto-detected format based on extension.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the data file to be loaded.
    **kwargs : dict
        Additional arguments passed to the specific format loader 
        (e.g., ``delimiter`` for CSV, ``target_column`` for ARFF).

    Returns
    -------
    Dataset
        Standardized dataset object containing data and metadata.

    Examples
    --------
    >>> from tuiml.datasets import load
    >>> data = load('iris.arff')
    >>> X, y = load('data.csv', target_column=-1)

    Notes
    -----
    **Supported Formats:**
    
    - ``.arff`` — Attribute-Relation File Format (WEKA native)
    - ``.csv``, ``.tsv`` — Delimited text files
    - ``.npy``, ``.npz`` — NumPy binary formats
    - ``.xlsx``, ``.xls`` — Microsoft Excel spreadsheets
    - ``.parquet``, ``.pq`` — Apache Parquet columnar storage
    - ``.json``, ``.jsonl`` — JSON and line-delimited JSON
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix not in LOADERS:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported: {list(LOADERS.keys())}"
        )

    loader = LOADERS[suffix]
    return loader(filepath, **kwargs)

def save(
    filepath: Union[str, Path],
    data: np.ndarray,
    target: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    **kwargs
):
    """Save data to a file with auto-detected format based on extension.

    Parameters
    ----------
    filepath : Union[str, Path]
        Output path where the file will be saved.
    data : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    target : np.ndarray or None, default=None
        Target values (labels) to include.
    feature_names : List[str] or None, default=None
        Names for the features.
    **kwargs : dict
        Additional arguments passed to the specific format saver.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.datasets import save
    >>> X = np.random.rand(100, 4)
    >>> save('my_data.parquet', X, feature_names=['f1', 'f2', 'f3', 'f4'])
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix not in SAVERS:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported: {list(SAVERS.keys())}"
        )

    saver = SAVERS[suffix]
    saver(filepath, data=data, target=target, feature_names=feature_names, **kwargs)
