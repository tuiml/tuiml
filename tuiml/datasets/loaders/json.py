"""
JSON file loader.

Load data from JSON files in various formats.
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from tuiml.datasets.loaders.arff import Dataset
from tuiml.datasets.loaders.pandas import load_pandas

def load_json(
    filepath: Union[str, Path],
    target_column: Optional[Union[str, int]] = -1,
    orient: str = 'auto',
    lines: bool = False,
    handle_categorical: str = 'encode'
) -> Dataset:
    """
    Load data from JSON file.

    Parameters
    ----------
    filepath : str or Path
        Path to JSON file
    target_column : str, int, or None
        Column name or index for target variable
        (None for unsupervised, -1 for last column)
    orient : str
        JSON structure format:
        - 'auto': Auto-detect format
        - 'records': [{col1: val1, col2: val2}, ...]
        - 'columns': {col1: [val1, val2], col2: [val1, val2]}
        - 'index': {idx1: {col1: val1}, idx2: {col1: val1}}
        - 'values': [[val1, val2], [val1, val2]]
        - 'split': {index: [], columns: [], data: []}
        - 'table': {schema: {}, data: []}
    lines : bool
        Whether file is JSON Lines format (one JSON object per line)
    handle_categorical : str
        How to handle categorical columns:
        - 'encode': Label encode to integers
        - 'drop': Drop categorical columns
        - 'error': Raise error if categorical found

    Returns
    -------
    result : Dataset
        Dataset object with X, y, feature_names

    Examples
    --------
    >>> data = load_json('data.json', target_column='class')
    >>> data.X.shape, data.y.shape

    >>> # JSON Lines format
    >>> data = load_json('data.jsonl', lines=True)

    >>> # Specific orientation
    >>> data = load_json('data.json', orient='records')
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for load_json. "
            "Install with: pip install pandas"
        )

    filepath = Path(filepath)

    # Auto-detect JSON Lines by extension
    if filepath.suffix.lower() in ('.jsonl', '.ndjson'):
        lines = True

    if lines:
        df = pd.read_json(filepath, lines=True)
    elif orient == 'auto':
        # Try to auto-detect format
        df = _load_json_auto(filepath)
    else:
        df = pd.read_json(filepath, orient=orient)

    df.name = filepath.stem

    return load_pandas(
        df,
        target_column=target_column,
        handle_categorical=handle_categorical
    )

def _load_json_auto(filepath: Path):
    """Auto-detect the JSON structure and load it as a DataFrame.

    Inspects the top-level type of the parsed JSON to determine the
    appropriate pandas orientation (records, columns, index, split, or
    table) and returns the corresponding DataFrame.

    Parameters
    ----------
    filepath : Path
        Path to the JSON file.

    Returns
    -------
    pandas.DataFrame
        Loaded data as a DataFrame.
    """
    import pandas as pd

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            # records format: [{}, {}, ...]
            return pd.DataFrame(data)
        elif len(data) > 0 and isinstance(data[0], list):
            # values format: [[], [], ...]
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(data)

    elif isinstance(data, dict):
        # Check for special formats
        if 'data' in data and 'columns' in data:
            # split or table format
            return pd.read_json(filepath, orient='split')
        elif 'schema' in data and 'data' in data:
            return pd.read_json(filepath, orient='table')
        else:
            # Could be columns or index format
            first_value = next(iter(data.values()))
            if isinstance(first_value, list):
                # columns format: {col: [values]}
                return pd.DataFrame(data)
            elif isinstance(first_value, dict):
                # index format: {idx: {col: val}}
                return pd.DataFrame.from_dict(data, orient='index')
            else:
                # Single row as dict
                return pd.DataFrame([data])

    raise ValueError(f"Unable to parse JSON format: {type(data)}")

def save_json(
    filepath: Union[str, Path],
    data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    target: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    target_column_name: str = "target",
    orient: str = 'records',
    lines: bool = False,
    indent: Optional[int] = 2
):
    """
    Save data to JSON format.

    Parameters
    ----------
    filepath : str or Path
        Output file path
    data : numpy.ndarray
        Feature data (n_samples, n_features)
    feature_names : list of str or None
        List of feature names
    target : numpy.ndarray or None
        Target values (optional)
    target_names : list of str or None
        Names of target classes (for classification)
    target_column_name : str
        Name for target column
    orient : str
        JSON structure format ('records', 'columns', 'values', 'split', 'table')
    lines : bool
        Whether to write as JSON Lines format
    indent : int or None
        Indentation level (None for compact, set when lines=False)

    Examples
    --------
    >>> save_json('output.json', X, feature_names=['a', 'b'], target=y)
    >>> save_json('output.jsonl', X, lines=True)
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for save_json. "
            "Install with: pip install pandas"
        )

    filepath = Path(filepath)

    if feature_names is None:
        feature_names = [f"col{i}" for i in range(data.shape[1])]

    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)

    # Add target column
    if target is not None:
        if target_names is not None:
            target_col = [target_names[int(i)] if not np.isnan(i) else None
                          for i in target]
        else:
            target_col = target
        df[target_column_name] = target_col

    # Write to JSON
    if lines:
        df.to_json(filepath, orient='records', lines=True)
    else:
        df.to_json(filepath, orient=orient, indent=indent)

def load_jsonl(
    filepath: Union[str, Path],
    target_column: Optional[Union[str, int]] = -1,
    handle_categorical: str = 'encode',
    max_lines: Optional[int] = None
) -> Dataset:
    """
    Load data from JSON Lines file.

    Each line is a separate JSON object (records format).

    Parameters
    ----------
    filepath : str or Path
        Path to JSONL file
    target_column : str, int, or None
        Column name or index for target variable
    handle_categorical : str
        How to handle categorical columns
    max_lines : int or None
        Maximum number of lines to read (None for all)

    Returns
    -------
    result : Dataset
        Dataset object

    Examples
    --------
    >>> data = load_jsonl('data.jsonl', target_column='label')
    >>> data = load_jsonl('large_data.jsonl', max_lines=10000)
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for load_jsonl. "
            "Install with: pip install pandas"
        )

    filepath = Path(filepath)

    if max_lines is not None:
        # Read limited lines
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                if line.strip():
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
    else:
        df = pd.read_json(filepath, lines=True)

    df.name = filepath.stem

    return load_pandas(
        df,
        target_column=target_column,
        handle_categorical=handle_categorical
    )

def save_jsonl(
    filepath: Union[str, Path],
    data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    target: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    target_column_name: str = "target"
):
    """
    Save data to JSON Lines format.

    Parameters
    ----------
    filepath : str or Path
        Output file path
    data : numpy.ndarray
        Feature data (n_samples, n_features)
    feature_names : list of str or None
        List of feature names
    target : numpy.ndarray or None
        Target values (optional)
    target_names : list of str or None
        Names of target classes
    target_column_name : str
        Name for target column

    Examples
    --------
    >>> save_jsonl('output.jsonl', X, feature_names=['a', 'b'], target=y)
    """
    save_json(
        filepath,
        data,
        feature_names=feature_names,
        target=target,
        target_names=target_names,
        target_column_name=target_column_name,
        lines=True
    )

def load_json_nested(
    filepath: Union[str, Path],
    record_path: Union[str, List[str]],
    meta: Optional[List[str]] = None,
    target_column: Optional[Union[str, int]] = -1,
    handle_categorical: str = 'encode'
) -> Dataset:
    """
    Load data from nested JSON structure.

    Flattens nested JSON using pandas json_normalize.

    Parameters
    ----------
    filepath : str or Path
        Path to JSON file
    record_path : str or list of str
        Path to records in nested structure (e.g., ['data', 'items'])
    meta : list of str or None
        Fields to include from parent levels
    target_column : str, int, or None
        Column name or index for target variable
    handle_categorical : str
        How to handle categorical columns

    Returns
    -------
    result : Dataset
        Dataset object

    Examples
    --------
    >>> # For JSON like: {"response": {"data": [{"a": 1}, {"a": 2}]}}
    >>> data = load_json_nested('api_response.json',
    ...                         record_path=['response', 'data'])
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for load_json_nested. "
            "Install with: pip install pandas"
        )

    filepath = Path(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    df = pd.json_normalize(json_data, record_path=record_path, meta=meta)
    df.name = filepath.stem

    return load_pandas(
        df,
        target_column=target_column,
        handle_categorical=handle_categorical
    )
