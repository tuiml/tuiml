"""Chebyshev (L-infinity) distance function."""

import numpy as np

def chebyshev_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    r"""Compute **Chebyshev (L-infinity)** distance between two points.

    The Chebyshev distance is the **maximum absolute difference** across all
    dimensions. Also known as **chessboard distance** or L-infinity norm.

    Theory
    ------
    .. math::
        d(x, y) = \max_i |x_i - y_i|

    This is the limiting case of the Minkowski distance as :math:`p \\to \infty`.

    Parameters
    ----------
    x1 : np.ndarray of shape (n_features,)
        First point.
    x2 : np.ndarray of shape (n_features,)
        Second point.

    Returns
    -------
    dist : float
        Chebyshev distance.

    Notes
    -----
    **Complexity:**

    - Time: :math:`O(n)` where :math:`n` is the number of features.

    See Also
    --------
    :func:`euclidean_distance` : L2 distance metric.
    :func:`minkowski_distance` : Generalized Lp distance metric.

    Examples
    --------
    Compute distance between two 2D points:

    >>> x1 = np.array([0, 0])
    >>> x2 = np.array([3, 4])
    >>> chebyshev_distance(x1, x2)
    4.0
    """
    return np.max(np.abs(x1 - x2))

def chebyshev_pairwise(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """Compute pairwise Chebyshev distances.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples_X, n_features)
        First set of samples.
    Y : np.ndarray of shape (n_samples_Y, n_features), optional, default=None
        Second set of samples. Defaults to X.

    Returns
    -------
    dist_matrix : np.ndarray of shape (n_samples_X, n_samples_Y)
        Distance matrix.

    See Also
    --------
    :func:`chebyshev_distance` : Point-to-point Chebyshev distance.
    """
    if Y is None:
        Y = X

    X = np.asarray(X)
    Y = np.asarray(Y)

    n_X, n_Y = len(X), len(Y)
    distances = np.zeros((n_X, n_Y))
    for i in range(n_X):
        distances[i] = np.max(np.abs(X[i] - Y), axis=1)
    return distances
