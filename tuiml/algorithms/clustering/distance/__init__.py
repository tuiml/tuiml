"""Distance metrics for measuring similarity between data points."""

import numpy as np
from typing import Callable, Optional

# Individual distance functions
from tuiml.algorithms.clustering.distance.euclidean import (
    euclidean_distance,
    euclidean_pairwise,
)
from tuiml.algorithms.clustering.distance.manhattan import (
    manhattan_distance,
    manhattan_pairwise,
)
from tuiml.algorithms.clustering.distance.cosine import (
    cosine_distance,
    cosine_pairwise,
)
from tuiml.algorithms.clustering.distance.chebyshev import (
    chebyshev_distance,
    chebyshev_pairwise,
)
from tuiml.algorithms.clustering.distance.minkowski import (
    minkowski_distance,
    minkowski_pairwise,
)

# Registry of distance functions
DISTANCE_FUNCTIONS = {
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance,
    'cosine': cosine_distance,
    'chebyshev': chebyshev_distance,
    'l1': manhattan_distance,
    'l2': euclidean_distance,
}

# Registry of pairwise distance functions (optimized)
PAIRWISE_FUNCTIONS = {
    'euclidean': euclidean_pairwise,
    'manhattan': manhattan_pairwise,
    'cosine': cosine_pairwise,
    'chebyshev': chebyshev_pairwise,
    'l1': manhattan_pairwise,
    'l2': euclidean_pairwise,
}

def get_distance_function(name: str) -> Callable:
    """Get a distance function by name.

    Parameters
    ----------
    name : str
        Name of the distance function 
        ('euclidean', 'manhattan', 'cosine', 'chebyshev', 'l1', 'l2').

    Returns
    -------
    dist_fn : Callable
        Distance function that takes two points and returns a scalar.

    Raises
    ------
    ValueError
        If the distance function name is unknown.

    Examples
    --------
    >>> dist_fn = get_distance_function('euclidean')
    >>> dist_fn(np.array([0, 0]), np.array([3, 4]))
    5.0
    """
    if name not in DISTANCE_FUNCTIONS:
        raise ValueError(
            f"Unknown distance function: {name}. "
            f"Available: {list(DISTANCE_FUNCTIONS.keys())}"
        )
    return DISTANCE_FUNCTIONS[name]

def pairwise_distances(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    metric: str = 'euclidean'
) -> np.ndarray:
    """Compute pairwise distances between samples.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples_X, n_features)
        First set of samples.
    Y : np.ndarray of shape (n_samples_Y, n_features), optional, default=None
        Second set of samples. If None, Y = X.
    metric : str, default='euclidean'
        Distance metric name ('euclidean', 'manhattan', 'cosine', etc.).

    Returns
    -------
    dist_matrix : np.ndarray of shape (n_samples_X, n_samples_Y)
        Distance matrix.

    Examples
    --------
    >>> X = np.array([[0, 0], [1, 1], [2, 2]])
    >>> pairwise_distances(X, metric='euclidean')
    array([[0.        , 1.41421356, 2.82842712],
           [1.41421356, 0.        , 1.41421356],
           [2.82842712, 1.41421356, 0.        ]])
    """
    if metric in PAIRWISE_FUNCTIONS:
        return PAIRWISE_FUNCTIONS[metric](X, Y)

    # Fallback: try to vectorize using numpy broadcasting
    dist_func = get_distance_function(metric)

    if Y is None:
        Y = X

    X = np.asarray(X)
    Y = np.asarray(Y)

    n_X, n_Y = len(X), len(Y)

    # Try vectorized computation using broadcasting
    # X[:, np.newaxis, :] has shape (n_X, 1, n_features)
    # Y[np.newaxis, :, :] has shape (1, n_Y, n_features)
    # Difference has shape (n_X, n_Y, n_features)
    try:
        diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]

        # Apply common distance computations vectorized
        if metric in ('euclidean', 'l2'):
            distances = np.sqrt(np.sum(diff ** 2, axis=2))
        elif metric in ('manhattan', 'l1'):
            distances = np.sum(np.abs(diff), axis=2)
        elif metric == 'chebyshev':
            distances = np.max(np.abs(diff), axis=2)
        else:
            # For truly custom metrics, use vectorized apply along axis
            # This is still faster than pure Python loops
            distances = np.array([[dist_func(X[i], Y[j]) for j in range(n_Y)] for i in range(n_X)])
    except (MemoryError, ValueError):
        # Fallback for very large arrays - use chunked computation
        distances = np.zeros((n_X, n_Y))
        chunk_size = 1000
        for i_start in range(0, n_X, chunk_size):
            i_end = min(i_start + chunk_size, n_X)
            for j_start in range(0, n_Y, chunk_size):
                j_end = min(j_start + chunk_size, n_Y)
                X_chunk = X[i_start:i_end]
                Y_chunk = Y[j_start:j_end]
                diff = X_chunk[:, np.newaxis, :] - Y_chunk[np.newaxis, :, :]
                distances[i_start:i_end, j_start:j_end] = np.sqrt(np.sum(diff ** 2, axis=2))

    return distances

def cdist(X: np.ndarray, Y: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """Compute distance between each pair of the two collections of inputs.

    Alias for `pairwise_distances` (SciPy compatibility).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples_X, n_features)
        First collection.
    Y : np.ndarray of shape (n_samples_Y, n_features)
        Second collection.
    metric : str, default='euclidean'
        Distance metric.

    Returns
    -------
    dist_matrix : np.ndarray of shape (n_samples_X, n_samples_Y)
        Distance matrix.
    """
    return pairwise_distances(X, Y, metric)

def pdist(X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """Compute pairwise distances between observations (condensed form).

    Returns a condensed distance matrix (upper triangular, flattened).
    Compatible with `scipy.spatial.distance.pdist`.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Observations.
    metric : str, default='euclidean'
        Distance metric.

    Returns
    -------
    dist_vector : np.ndarray of shape (n_samples * (n_samples - 1) / 2,)
        Condensed distance matrix.
    """
    n = len(X)
    dist_func = get_distance_function(metric)
    distances = []

    for i in range(n):
        for j in range(i + 1, n):
            distances.append(dist_func(X[i], X[j]))

    return np.array(distances)

__all__ = [
    # Individual distance functions
    "euclidean_distance",
    "manhattan_distance",
    "cosine_distance",
    "chebyshev_distance",
    "minkowski_distance",
    # Pairwise functions
    "euclidean_pairwise",
    "manhattan_pairwise",
    "cosine_pairwise",
    "chebyshev_pairwise",
    "minkowski_pairwise",
    # Utilities
    "get_distance_function",
    "pairwise_distances",
    "cdist",
    "pdist",
    # Registries
    "DISTANCE_FUNCTIONS",
    "PAIRWISE_FUNCTIONS",
]
