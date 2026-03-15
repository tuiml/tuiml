"""Cosine distance function."""

import numpy as np

def cosine_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    r"""Compute **cosine distance** between two points.

    Cosine distance measures the **angular dissimilarity** between two vectors,
    defined as :math:`1 - \\text{cosine\_similarity}`.

    Theory
    ------
    .. math::
        d(x, y) = 1 - \\frac{x \cdot y}{\|x\|_2 \, \|y\|_2}

    **Score interpretation:**

    - :math:`d = 0` -- Vectors have the same direction
    - :math:`d = 1` -- Vectors are orthogonal
    - :math:`d = 2` -- Vectors point in opposite directions

    Parameters
    ----------
    x1 : np.ndarray of shape (n_features,)
        First point.
    x2 : np.ndarray of shape (n_features,)
        Second point.

    Returns
    -------
    dist : float
        Cosine distance in the range [0, 2].

    Notes
    -----
    **Complexity:**

    - Time: :math:`O(n)` where :math:`n` is the number of features.

    See Also
    --------
    :func:`euclidean_distance` : L2 distance metric (magnitude-sensitive).

    Examples
    --------
    Orthogonal vectors have cosine distance 1.0:

    >>> x1 = np.array([1, 0])
    >>> x2 = np.array([0, 1])
    >>> cosine_distance(x1, x2)
    1.0
    """
    dot = np.dot(x1, x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return 1.0 - dot / (norm1 * norm2)

def cosine_pairwise(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """Compute pairwise cosine distances.

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
    :func:`cosine_distance` : Point-to-point cosine distance.
    """
    if Y is None:
        Y = X

    X = np.asarray(X)
    Y = np.asarray(Y)

    # Normalize vectors
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)

    # Handle zero vectors
    X_norm = np.where(X_norm == 0, 1, X_norm)
    Y_norm = np.where(Y_norm == 0, 1, Y_norm)

    X_normalized = X / X_norm
    Y_normalized = Y / Y_norm

    # Cosine similarity -> distance
    similarity = X_normalized @ Y_normalized.T
    return 1.0 - similarity
