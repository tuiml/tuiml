"""
Excel spreadsheet loader.

Reads ``.xlsx`` and ``.xls`` workbooks into a
:class:`~tuiml.datasets.loaders.arff.Dataset`, one sheet at a time or all of
them at once. Column typing and target extraction are delegated to
:func:`~tuiml.datasets.loaders.pandas.load_pandas`, so categorical columns are
handled the same way as for any other tabular source.

Requires ``pandas`` and ``openpyxl``; both are imported lazily, so the rest of
``tuiml.datasets.loaders`` works without them.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Union

from tuiml.datasets.loaders.arff import Dataset
from tuiml.datasets.loaders.pandas import load_pandas

def load_excel(
    filepath: Union[str, Path],
    target_column: Optional[Union[str, int]] = -1,
    sheet_name: Union[str, int] = 0,
    handle_categorical: str = 'encode'
) -> Dataset:
    """Load a single sheet from an Excel workbook.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.xlsx`` or ``.xls`` file.
    target_column : str, int, or None, default=-1
        The column to treat as the target variable:

        - ``-1``: Use the last column
        - ``int``: Use specific zero-based index
        - ``str``: Use column name
        - ``None``: Do not extract a target (X will contain all columns)
    sheet_name : str or int, default=0
        Sheet to read, either by name or by zero-based position.
    handle_categorical : str, default='encode'
        What to do with non-numeric columns: ``'encode'`` maps each category to
        an integer, ``'drop'`` removes the column, ``'error'`` raises.

    Returns
    -------
    result : Dataset
        Standardized dataset object containing data and metadata.

    Raises
    ------
    ImportError
        If ``pandas`` (and ``openpyxl``) are not installed.

    Examples
    --------
    >>> from tuiml.datasets.loaders import load_excel
    >>> data = load_excel('sales.xlsx', sheet_name='Q1')
    >>> X, y = data
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for load_excel. "
            "Install with: pip install pandas openpyxl"
        )
        
    filepath = Path(filepath)
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df.name = filepath.stem
    
    return load_pandas(
        df, 
        target_column=target_column,
        handle_categorical=handle_categorical
    )

def save_excel(
    filepath: Union[str, Path],
    data: np.ndarray,
    feature_names: Optional[List[str]] = None,
    target: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    target_column_name: str = "target",
    sheet_name: str = "Sheet1"
):
    """Write features and an optional target to an Excel workbook.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    data : numpy.ndarray of shape (n_samples, n_features)
        Feature matrix to write.
    feature_names : list of str or None, default=None
        Column headers. Defaults to ``col0``, ``col1``, ...
    target : numpy.ndarray of shape (n_samples,) or None, default=None
        Target values. Appended as an extra column when given.
    target_names : list of str or None, default=None
        Class names. When given, integer targets are written as these labels
        instead of numbers.
    target_column_name : str, default='target'
        Header for the target column.
    sheet_name : str, default='Sheet1'
        Name of the sheet to write.

    Returns
    -------
    None
        The workbook is written to ``filepath``.

    Raises
    ------
    ImportError
        If ``pandas`` (and ``openpyxl``) are not installed.

    Examples
    --------
    >>> from tuiml.datasets.loaders import save_excel
    >>> save_excel('out.xlsx', X, feature_names=['a', 'b'], target=y)
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for save_excel. "
            "Install with: pip install pandas openpyxl"
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
        
    df.to_excel(filepath, sheet_name=sheet_name, index=False)

def load_excel_sheets(
    filepath: Union[str, Path],
    target_column: Optional[Union[str, int]] = -1,
    handle_categorical: str = 'encode'
) -> dict:
    """Load every sheet of an Excel workbook as a separate dataset.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.xlsx`` or ``.xls`` file.
    target_column : str, int, or None, default=-1
        The column to treat as the target variable, applied to every sheet.
        See :func:`load_excel` for the accepted values.
    handle_categorical : str, default='encode'
        What to do with non-numeric columns: ``'encode'``, ``'drop'``, or
        ``'error'``.

    Returns
    -------
    result : dict
        Dictionary mapping each sheet name to its Dataset.

    Raises
    ------
    ImportError
        If ``pandas`` (and ``openpyxl``) are not installed.

    Examples
    --------
    >>> from tuiml.datasets.loaders import load_excel_sheets
    >>> sheets = load_excel_sheets('report.xlsx')
    >>> sorted(sheets)
    ['Q1', 'Q2']
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for load_excel_sheets. "
            "Install with: pip install pandas openpyxl"
        )

    filepath = Path(filepath)
    excel_file = pd.ExcelFile(filepath)

    datasets = {}
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        df.name = f"{filepath.stem}_{sheet_name}"
        datasets[sheet_name] = load_pandas(
            df,
            target_column=target_column,
            handle_categorical=handle_categorical
        )

    return datasets
