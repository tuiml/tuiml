"""
NumPy format loader (.npy, .npz).
"""

import numpy as np
from typing import List, Optional, Union
from pathlib import Path

from tuiml.datasets.loaders.arff import Dataset

def load_numpy(
    filepath: Union[str, Path],
    target_key: str = 'y',
    data_key: str = 'X'
) -> Dataset:
    """
    Load data from NumPy format (.npy or .npz).

    For .npy files: assumes the file contains feature data only.
    For .npz files: looks for 'X' (features) and 'y' (target) arrays.

    Parameters
    ----------
    filepath : str or Path
        Path to .npy or .npz file
    target_key : str
        Key for target array in .npz file
    data_key : str
        Key for data array in .npz file

    Returns
    -------
    result : Dataset
        Dataset object with X, y, feature_names

    Examples
    --------
    >>> data = load_numpy('data.npz')
    >>> data.X.shape, data.y.shape
    >>> X, y = load_numpy('data.npz')  # Can unpack
    """
    filepath = Path(filepath)

    if filepath.suffix == '.npy':
        data = np.load(filepath)
        target = None
    elif filepath.suffix == '.npz':
        archive = np.load(filepath)
        data = archive[data_key] if data_key in archive else archive['data']
        target = archive[target_key] if target_key in archive else None
    else:
        # Try to load as generic numpy
        data = np.load(filepath, allow_pickle=True)
        target = None

    return Dataset(
        X=data,
        y=target,
        feature_names=[f"feat{i}" for i in range(data.shape[1])],
        name=filepath.stem
    )

def save_numpy(
    filepath: Union[str, Path],
    data: np.ndarray,
    target: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    compressed: bool = True
):
    """
    Save data to NumPy format.

    If target is provided, saves as .npz with 'X' and 'y' keys.
    Otherwise saves as .npy.

    Parameters
    ----------
    filepath : str or Path
        Output file path
    data : numpy.ndarray
        Feature data (n_samples, n_features)
    target : numpy.ndarray or None
        Target values (optional)
    feature_names : list of str or None
        List of feature names (stored in .npz)
    compressed : bool
        Whether to use compression (.npz only)
    """
    filepath = Path(filepath)

    if target is None and feature_names is None:
        np.save(filepath, data)
    else:
        save_dict = {'X': data}
        if target is not None:
            save_dict['y'] = target
        if feature_names is not None:
            save_dict['feature_names'] = np.array(feature_names)

        if compressed:
            np.savez_compressed(filepath, **save_dict)
        else:
            np.savez(filepath, **save_dict)
