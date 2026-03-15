"""Euclidean (L2) distance function."""

import numpy as np

def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    r"""Compute **Euclidean (L2)** distance between two points.

    The Euclidean distance is the **straight-line distance** between two points
    in Euclidean space.

    Theory
    ------
    .. math::
        d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}

    This is a special case of the Minkowski distance with :math:`p = 2`.

    Parameters
    ----------
    x1 : np.ndarray of shape (n_features,)
        First point.
    x2 : np.ndarray of shape (n_features,)
        Second point.

    Returns
    -------
    dist : float
        Euclidean distance.

    Notes
    -----
    **Complexity:**

    - Time: :math:`O(n)` where :math:`n` is the number of features.

    See Also
    --------
    :func:`manhattan_distance` : L1 distance metric.
    :func:`minkowski_distance` : Generalized Lp distance metric.

    Examples
    --------
    Compute distance between two 2D points:

    >>> x1 = np.array([0, 0])
    >>> x2 = np.array([3, 4])
    >>> euclidean_distance(x1, x2)
    5.0
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

def euclidean_pairwise(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    r"""Compute pairwise Euclidean distances efficiently using vectorization.

    Uses the algebraic identity to avoid explicit broadcasting:

    .. math::
        \|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2 \langle x, y \\rangle

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
    :func:`euclidean_distance` : Point-to-point Euclidean distance.
    """
    if Y is None:
        Y = X

    X = np.asarray(X)
    Y = np.asarray(Y)

    XX = np.sum(X ** 2, axis=1)[:, np.newaxis]
    YY = np.sum(Y ** 2, axis=1)[np.newaxis, :]
    distances = XX + YY - 2 * X @ Y.T
    distances = np.maximum(distances, 0)  # Handle numerical errors
    return np.sqrt(distances)
