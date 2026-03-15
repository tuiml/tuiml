"""
JSON file loader.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Union

from tuiml.datasets.loaders.arff import Dataset
from tuiml.datasets.loaders.pandas import load_pandas

def load_json(
    filepath: Union[str, Path],
    target_column: Optional[Union[str, int]] = -1,
    orient: str = 'records',
    lines: bool = False,
    handle_categorical: str = 'encode'
) -> Dataset:
    """Load data from a JSON file into a Dataset.

    Parameters
    ----------
    filepath : str or Path
        Path to the JSON file.
    target_column : str, int, or None, default=-1
        Column name or index to use as target. Use -1 for the last column,
        or None to skip target extraction.
    orient : str, default='records'
        Expected JSON orientation. One of ``'records'``, ``'split'``,
        ``'index'``, ``'columns'``, or ``'values'``.
    lines : bool, default=False
        If True, read the file as JSON Lines (one object per line).
    handle_categorical : str, default='encode'
        How to handle categorical columns: ``'encode'``, ``'drop'``,
        or ``'error'``.

    Returns
    -------
    Dataset
        Dataset object containing features, target, and metadata.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for load_json. "
            "Install with: pip install pandas"
        )
        
    filepath = Path(filepath)
    df = pd.read_json(filepath, orient=orient, lines=lines)
    df.name = filepath.stem
    
    return load_pandas(
        df, 
        target_column=target_column,
        handle_categorical=handle_categorical
    )

def save_json(
    filepath: Union[str, Path],
    data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    target: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    target_column_name: str = "target",
    orient: str = 'records',
    lines: bool = False
):
    """Save data to a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    data : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    feature_names : list of str or None, default=None
        Names for each feature column. If None, generic names are used.
    target : np.ndarray or None, default=None
        Target values to include in the output.
    target_names : list of str or None, default=None
        Class names for nominal targets.
    target_column_name : str, default="target"
        Column name for the target in the output JSON.
    orient : str, default='records'
        JSON orientation for pandas ``to_json``.
    lines : bool, default=False
        If True, write as JSON Lines format.
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
        
    df = pd.DataFrame(data, columns=feature_names)
    
    if target is not None:
        if target_names is not None:
            target_col = [target_names[int(i)] if not np.isnan(i) else None 
                          for i in target]
        else:
            target_col = target
        df[target_column_name] = target_col
        
    df.to_json(filepath, orient=orient, lines=lines, index=False)
