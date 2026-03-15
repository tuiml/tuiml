"""
CSV (Comma-Separated Values) loader.
"""

import numpy as np
from typing import List, Optional, Union
from pathlib import Path

from tuiml.datasets.loaders.arff import Dataset

def load_csv(
    filepath: Union[str, Path],
    target_column: Union[int, str] = -1,
    header: bool = True,
    delimiter: str = ','
) -> Dataset:
    """Load data from CSV (Comma-Separated Values) files.

    Standard CSV loader supporting headers, custom delimiters, and automatic
    target extraction.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the CSV file to be loaded.
    target_column : int or str, default=-1
        The column to treat as the target variable:

        - ``-1`` — Use the last column
        - ``int`` — Use specific zero-based index
        - ``str`` — Use column name (requires header=True)
        - ``None`` — Do not extract a target (X will contain all columns)
    header : bool, default=True
        Whether the first row of the CSV contains column names.
    delimiter : str, default=','
        The character used to separate columns in the file.

    Returns
    -------
    Dataset
        Standardized dataset object containing data and metadata.

    Examples
    --------
    >>> from tuiml.datasets.loaders import load_csv
    >>> data = load_csv('data.csv')
    >>> X, y = data
    >>> print(X.shape)
    """
    filepath = Path(filepath)

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    if not lines:
        raise ValueError("Empty file")

    # Parse header
    if header:
        header_line = lines[0].strip()
        feature_names = [n.strip().strip('"\'') for n in header_line.split(delimiter)]
        data_lines = lines[1:]
    else:
        data_lines = lines
        feature_names = None

    # Parse data - keep raw string values first
    raw_rows = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        values = [val.strip().strip('"\'') for val in line.split(delimiter)]
        raw_rows.append(values)

    if not raw_rows:
        raise ValueError("No data rows found")

    n_cols = len(raw_rows[0])

    if feature_names is None:
        feature_names = [f"col{i}" for i in range(n_cols)]

    # Resolve target_column if string
    target_idx = None
    if target_column is not None:
        if isinstance(target_column, str):
            if target_column in feature_names:
                target_idx = feature_names.index(target_column)
            else:
                raise ValueError(f"Target column '{target_column}' not found in headers: {feature_names}")
        else:
            target_idx = target_column
            if target_idx < 0:
                target_idx = n_cols + target_idx

    # Extract target column (keep as strings for categorical)
    target = None
    target_raw = None
    if target_idx is not None:
        target_raw = [row[target_idx] for row in raw_rows]

        # Check if target is categorical (non-numeric)
        is_categorical = False
        for val in target_raw:
            if val not in ('', 'na', 'nan', '?', 'null'):
                try:
                    float(val)
                except ValueError:
                    is_categorical = True
                    break

        if is_categorical:
            # Encode categorical target
            unique_classes = sorted(set(v for v in target_raw if v not in ('', 'na', 'nan', '?', 'null')))
            class_to_idx = {c: i for i, c in enumerate(unique_classes)}
            target = np.array([
                class_to_idx.get(v, -1) if v not in ('', 'na', 'nan', '?', 'null') else np.nan
                for v in target_raw
            ], dtype=float)
        else:
            # Numeric target
            target = np.array([
                float(v) if v not in ('', 'na', 'nan', '?', 'null') else np.nan
                for v in target_raw
            ], dtype=float)

    # Parse features (numeric only)
    feature_idx = [i for i in range(n_cols) if i != target_idx]
    features = []
    for row in raw_rows:
        feat_values = []
        for i in feature_idx:
            val = row[i]
            if val == '' or val.lower() in ('na', 'nan', '?', 'null'):
                feat_values.append(np.nan)
            else:
                try:
                    feat_values.append(float(val))
                except ValueError:
                    feat_values.append(np.nan)
        features.append(feat_values)

    X = np.array(features, dtype=float)
    names = [feature_names[i] for i in feature_idx]

    return Dataset(
        X=X,
        y=target,
        feature_names=names,
        name=filepath.stem
    )

def save_csv(
    filepath: Union[str, Path],
    data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    target: Optional[np.ndarray] = None,
    target_name: str = "target",
    delimiter: str = ','
):
    """Save data to CSV (Comma-Separated Values) format.

    Parameters
    ----------
    filepath : Union[str, Path]
        Output path where the CSV file will be saved.
    data : np.ndarray
        Feature matrix to save.
    feature_names : List[str] or None, default=None
        Names for the features. If None, generic names like ``col0``, ``col1`` 
        will be used.
    target : np.ndarray or None, default=None
        Target values (labels) to include in the output.
    target_name : str, default="target"
        The header name for the target column.
    delimiter : str, default=','
        The character used to separate columns in the file.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.datasets.loaders import save_csv
    >>> X = np.random.rand(10, 2)
    >>> save_csv('output.csv', X, feature_names=['f1', 'f2'])
    """
    filepath = Path(filepath)

    if feature_names is None:
        feature_names = [f"col{i}" for i in range(data.shape[1])]

    with open(filepath, 'w') as f:
        # Header
        header = delimiter.join(feature_names)
        if target is not None:
            header += f"{delimiter}{target_name}"
        f.write(header + "\n")

        # Data
        for i in range(len(data)):
            row = delimiter.join(str(v) for v in data[i])
            if target is not None:
                row += f"{delimiter}{target[i]}"
            f.write(row + "\n")
