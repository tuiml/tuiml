"""Minkowski distance function."""

import numpy as np

def minkowski_distance(x1: np.ndarray, x2: np.ndarray, p: float = 2) -> float:
    r"""Compute **Minkowski** distance between two points.

    The Minkowski distance is a **generalized metric** that encompasses
    :math:`L_1`, :math:`L_2`, and :math:`L_\infty` distances as special cases.

    Theory
    ------
    .. math::
        d(x, y) = \left( \sum_{i=1}^n |x_i - y_i|^p \\right)^{1/p}

    **Special cases:**

    - :math:`p=1`: Manhattan distance
    - :math:`p=2`: Euclidean distance
    - :math:`p \\to \infty`: Chebyshev distance

    Parameters
    ----------
    x1 : np.ndarray of shape (n_features,)
        First point.
    x2 : np.ndarray of shape (n_features,)
        Second point.
    p : float, default=2
        Order of the norm.

    Returns
    -------
    dist : float
        Minkowski distance.

    Notes
    -----
    **Complexity:**

    - Time: :math:`O(n)` where :math:`n` is the number of features.

    See Also
    --------
    :func:`euclidean_distance` : Minkowski distance with :math:`p=2`.
    :func:`manhattan_distance` : Minkowski distance with :math:`p=1`.
    :func:`chebyshev_distance` : Minkowski distance with :math:`p=\infty`.

    Examples
    --------
    Euclidean distance (p=2) and Manhattan distance (p=1):

    >>> x1 = np.array([0, 0])
    >>> x2 = np.array([3, 4])
    >>> minkowski_distance(x1, x2, p=2)
    5.0
    >>> minkowski_distance(x1, x2, p=1)
    7.0
    """
    if p == float('inf'):
        return np.max(np.abs(x1 - x2))
    return np.power(np.sum(np.abs(x1 - x2) ** p), 1.0 / p)

def minkowski_pairwise(X: np.ndarray, Y: np.ndarray = None, p: float = 2) -> np.ndarray:
    """Compute pairwise Minkowski distances.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples_X, n_features)
        First set of samples.
    Y : np.ndarray of shape (n_samples_Y, n_features), optional, default=None
        Second set of samples. Defaults to X.
    p : float, default=2
        Order of the norm.

    Returns
    -------
    dist_matrix : np.ndarray of shape (n_samples_X, n_samples_Y)
        Distance matrix.

    See Also
    --------
    :func:`minkowski_distance` : Point-to-point Minkowski distance.
    """
    if Y is None:
        Y = X

    X = np.asarray(X)
    Y = np.asarray(Y)

    n_X, n_Y = len(X), len(Y)
    distances = np.zeros((n_X, n_Y))

    if p == float('inf'):
        for i in range(n_X):
            distances[i] = np.max(np.abs(X[i] - Y), axis=1)
    else:
        for i in range(n_X):
            distances[i] = np.power(
                np.sum(np.abs(X[i] - Y) ** p, axis=1),
                1.0 / p
            )
    return distances
