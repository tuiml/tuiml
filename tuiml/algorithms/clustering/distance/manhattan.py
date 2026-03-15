"""Manhattan (L1) distance function."""

import numpy as np

def manhattan_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    r"""Compute **Manhattan (L1)** distance between two points.

    The Manhattan distance is the sum of **absolute differences** of their
    coordinates. Also known as **city block** distance or taxicab distance.

    Theory
    ------
    .. math::
        d(x, y) = \sum_{i=1}^n |x_i - y_i|

    This is a special case of the Minkowski distance with :math:`p = 1`.

    Parameters
    ----------
    x1 : np.ndarray of shape (n_features,)
        First point.
    x2 : np.ndarray of shape (n_features,)
        Second point.

    Returns
    -------
    dist : float
        Manhattan distance.

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
    >>> manhattan_distance(x1, x2)
    7.0
    """
    return np.sum(np.abs(x1 - x2))

def manhattan_pairwise(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """Compute pairwise Manhattan distances.

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
    :func:`manhattan_distance` : Point-to-point Manhattan distance.
    """
    if Y is None:
        Y = X

    X = np.asarray(X)
    Y = np.asarray(Y)

    n_X, n_Y = len(X), len(Y)
    distances = np.zeros((n_X, n_Y))
    for i in range(n_X):
        distances[i] = np.sum(np.abs(X[i] - Y), axis=1)
    return distances
