"""
Pandas DataFrame loader.

Load data from pandas DataFrames with automatic type detection.
"""

import numpy as np
from typing import List, Optional, Union
from tuiml.datasets.loaders.arff import Dataset

def load_pandas(
    df,
    target_column: Optional[Union[str, int]] = None,
    feature_columns: Optional[List[Union[str, int]]] = None,
    drop_columns: Optional[List[Union[str, int]]] = None,
    handle_categorical: str = 'encode'
) -> Dataset:
    """
    Load data from pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        pandas DataFrame
    target_column : str, int, or None
        Column name or index for target variable
        (None for unsupervised, -1 for last column)
    feature_columns : list of str or int, or None
        List of columns to use as features
        (None to use all except target)
    drop_columns : list of str or int, or None
        List of columns to drop before processing
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
    >>> import pandas as pd
    >>> df = pd.read_csv('data.csv')
    >>> data = load_pandas(df, target_column='species')
    >>> data.X.shape, data.y.shape

    >>> # Using column index
    >>> data = load_pandas(df, target_column=-1)

    >>> # Select specific features
    >>> data = load_pandas(df, target_column='target',
    ...                    feature_columns=['age', 'income', 'score'])
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for load_pandas. "
            "Install with: pip install pandas"
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

    df = df.copy()

    # Drop specified columns
    if drop_columns is not None:
        cols_to_drop = []
        for col in drop_columns:
            if isinstance(col, int):
                cols_to_drop.append(df.columns[col])
            else:
                cols_to_drop.append(col)
        df = df.drop(columns=cols_to_drop)

    # Resolve target column
    target_col_name = None
    if target_column is not None:
        if isinstance(target_column, int):
            if target_column < 0:
                target_column = len(df.columns) + target_column
            target_col_name = df.columns[target_column]
        else:
            target_col_name = target_column

    # Determine feature columns
    if feature_columns is not None:
        feature_cols = []
        for col in feature_columns:
            if isinstance(col, int):
                feature_cols.append(df.columns[col])
            else:
                feature_cols.append(col)
    else:
        feature_cols = [c for c in df.columns if c != target_col_name]

    # Extract target
    if target_col_name is not None:
        target_series = df[target_col_name]
        target_names = None

        if target_series.dtype == 'object' or pd.api.types.is_categorical_dtype(target_series):
            # Encode categorical target
            categories = target_series.astype('category')
            target_names = list(categories.cat.categories)
            target = categories.cat.codes.values.astype(float)
            # Handle missing as NaN
            target[target == -1] = np.nan
        else:
            target = target_series.values.astype(float)
    else:
        target = None
        target_names = None

    # Process features
    feature_df = df[feature_cols].copy()
    feature_names = list(feature_cols)

    # Handle categorical columns
    categorical_cols = feature_df.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    if categorical_cols:
        if handle_categorical == 'encode':
            for col in categorical_cols:
                feature_df[col] = feature_df[col].astype('category').cat.codes
                # Replace -1 (missing) with NaN
                feature_df.loc[feature_df[col] == -1, col] = np.nan
        elif handle_categorical == 'drop':
            feature_df = feature_df.drop(columns=categorical_cols)
            feature_names = [c for c in feature_names if c not in categorical_cols]
        elif handle_categorical == 'error':
            raise ValueError(
                f"Categorical columns found: {categorical_cols}. "
                "Use handle_categorical='encode' or 'drop'"
            )

    # Convert to numpy
    X = feature_df.values.astype(float)

    # Get dataset name from DataFrame name attribute if available
    name = getattr(df, 'name', None) or 'dataframe'

    return Dataset(
        X=X,
        y=target,
        feature_names=feature_names,
        target_names=target_names,
        name=name
    )

def to_pandas(dataset: Dataset, include_target: bool = True):
    """
    Convert Dataset to pandas DataFrame.

    Parameters
    ----------
    dataset : Dataset
        Dataset object
    include_target : bool
        Whether to include target column

    Returns
    -------
    result : pandas.DataFrame
        pandas DataFrame

    Examples
    --------
    >>> data = load_arff('iris.arff')
    >>> df = to_pandas(data)
    >>> df.head()
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for to_pandas. "
            "Install with: pip install pandas"
        )

    df = pd.DataFrame(dataset.X, columns=dataset.feature_names)

    if include_target and dataset.y is not None:
        if dataset.target_names is not None:
            # Convert numeric target back to labels
            target_col = [dataset.target_names[int(i)] if not np.isnan(i) else None
                          for i in dataset.y]
        else:
            target_col = dataset.y
        df['target'] = target_col

    df.name = dataset.name
    return df

def from_pandas(
    df,
    target_column: Optional[Union[str, int]] = None,
    **kwargs
) -> Dataset:
    """
    Alias for load_pandas for consistent naming.

    Parameters
    ----------
    df : pandas.DataFrame
        pandas DataFrame
    target_column : str, int, or None
        Column name or index for target variable
    **kwargs : dict
        Additional arguments passed to load_pandas

    Returns
    -------
    result : Dataset
        Dataset object
    """
    return load_pandas(df, target_column=target_column, **kwargs)
