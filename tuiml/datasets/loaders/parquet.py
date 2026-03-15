"""
Parquet file loader.

Load data from Apache Parquet files using pyarrow or fastparquet.
"""

import numpy as np
from typing import List, Optional, Union
from pathlib import Path

from tuiml.datasets.loaders.arff import Dataset
from tuiml.datasets.loaders.pandas import load_pandas

def load_parquet(
    filepath: Union[str, Path],
    target_column: Optional[Union[str, int]] = -1,
    columns: Optional[List[str]] = None,
    handle_categorical: str = 'encode',
    engine: str = 'auto'
) -> Dataset:
    """
    Load data from Parquet file.

    Parameters
    ----------
    filepath : str or Path
        Path to Parquet file
    target_column : str, int, or None
        Column name or index for target variable
        (None for unsupervised, -1 for last column)
    columns : list of str or None
        List of columns to read (None for all)
    handle_categorical : str
        How to handle categorical columns:
        - 'encode': Label encode to integers
        - 'drop': Drop categorical columns
        - 'error': Raise error if categorical found
    engine : str
        Parquet engine ('auto', 'pyarrow', 'fastparquet')

    Returns
    -------
    result : Dataset
        Dataset object with X, y, feature_names

    Examples
    --------
    >>> data = load_parquet('data.parquet', target_column='class')
    >>> data.X.shape, data.y.shape

    >>> # Read specific columns
    >>> data = load_parquet('data.parquet', columns=['age', 'income', 'target'])
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for load_parquet. "
            "Install with: pip install pandas"
        )

    filepath = Path(filepath)

    # Determine engine
    if engine == 'auto':
        try:
            import pyarrow
            engine = 'pyarrow'
        except ImportError:
            try:
                import fastparquet
                engine = 'fastparquet'
            except ImportError:
                raise ImportError(
                    "Either pyarrow or fastparquet is required for Parquet files. "
                    "Install with: pip install pyarrow"
                )

    # Read parquet file
    df = pd.read_parquet(filepath, columns=columns, engine=engine)
    df.name = filepath.stem

    return load_pandas(
        df,
        target_column=target_column,
        handle_categorical=handle_categorical
    )

def save_parquet(
    filepath: Union[str, Path],
    data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    target: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    target_column_name: str = "target",
    compression: str = 'snappy',
    engine: str = 'auto'
):
    """
    Save data to Parquet format.

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
    compression : str or None
        Compression codec ('snappy', 'gzip', 'brotli', None)
    engine : str
        Parquet engine ('auto', 'pyarrow', 'fastparquet')

    Examples
    --------
    >>> save_parquet('output.parquet', X, feature_names=['a', 'b'], target=y)
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for save_parquet. "
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

    # Determine engine
    if engine == 'auto':
        try:
            import pyarrow
            engine = 'pyarrow'
        except ImportError:
            engine = 'fastparquet'

    # Write to parquet
    df.to_parquet(filepath, compression=compression, engine=engine, index=False)

def load_parquet_partitioned(
    directory: Union[str, Path],
    target_column: Optional[Union[str, int]] = -1,
    filters: Optional[List] = None,
    handle_categorical: str = 'encode',
    engine: str = 'auto'
) -> Dataset:
    """
    Load data from partitioned Parquet dataset.

    Parameters
    ----------
    directory : str or Path
        Path to partitioned parquet directory
    target_column : str, int, or None
        Column name or index for target variable
    filters : list or None
        Row group filters (e.g., [('col', '>', 5)])
    handle_categorical : str
        How to handle categorical columns
    engine : str
        Parquet engine

    Returns
    -------
    result : Dataset
        Dataset object

    Examples
    --------
    >>> # Load partitioned dataset with filtering
    >>> data = load_parquet_partitioned(
    ...     'data_partitioned/',
    ...     filters=[('year', '>=', 2020)]
    ... )
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for load_parquet_partitioned. "
            "Install with: pip install pandas"
        )

    directory = Path(directory)

    if engine == 'auto':
        try:
            import pyarrow
            engine = 'pyarrow'
        except ImportError:
            engine = 'fastparquet'

    # Read partitioned dataset
    df = pd.read_parquet(directory, engine=engine, filters=filters)
    df.name = directory.name

    return load_pandas(
        df,
        target_column=target_column,
        handle_categorical=handle_categorical
    )
