"""
Excel file loader.
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
    """
    Load data from Excel file.

    Parameters
    ----------
    filepath : str or Path
        Path to Excel file
    target_column : str, int, or None
        Target column name or index
    sheet_name : str or int
        Sheet to read (name or index)
    handle_categorical : str
        'encode', 'drop', or 'error'

    Returns
    -------
    result : Dataset
        Dataset object
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
    """
    Save data to Excel format.

    Parameters
    ----------
    filepath : str or Path
        Output file path
    data : numpy.ndarray
        Feature data
    feature_names : list of str or None
        Feature names
    target : numpy.ndarray or None
        Target values
    target_names : list of str or None
        Target class names
    target_column_name : str
        Name for target column
    sheet_name : str
        Sheet name
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
    """
    Load all sheets from an Excel file.

    Parameters
    ----------
    filepath : str or Path
        Path to Excel file
    target_column : str, int, or None
        Target column name or index
    handle_categorical : str
        'encode', 'drop', or 'error'

    Returns
    -------
    result : dict
        Dictionary mapping sheet names to Dataset objects
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
