"""Linear (Brute Force) Nearest Neighbor Search."""

import numpy as np
from typing import Tuple

from tuiml.base.neighbors import NearestNeighborSearch
from tuiml._cpp_ext import neighbors as _cpp_nn

class LinearNNSearch(NearestNeighborSearch):
    """Linear (brute force) nearest neighbor search via **exhaustive distance computation**.

    This algorithm computes the distance from the query point to **all** points
    in the training dataset. While simple and always exact, its query time
    scales linearly with the number of training samples, making it the
    **baseline** against which tree-based methods are compared.

    Overview
    --------
    The algorithm operates in the following steps:

    1. Store all training points during ``build()``.
    2. For a query, compute the **squared Euclidean distance** to every
       training point using a vectorized expansion:
       :math:`\\|q - x_i\\|^2 = \\|q\\|^2 + \\|x_i\\|^2 - 2 q^T x_i`.
    3. Use ``argpartition`` to efficiently find the :math:`k` smallest
       distances in :math:`O(n)` time.
    4. Sort only the :math:`k` selected neighbors by distance.

    Theory
    ------
    The distance metric is **Euclidean distance**:

    .. math::
        d(q, x_i) = \\sqrt{\\sum_{j=1}^{m} (q_j - x_{i,j})^2}

    Using the expansion :math:`\\|a - b\\|^2 = \\|a\\|^2 + \\|b\\|^2 - 2 a^T b`
    allows the computation to leverage optimized BLAS matrix-vector products,
    yielding a constant-factor speedup over a naive loop.

    Attributes
    ----------
    X_ : np.ndarray
        The training data stored for search.
    n_samples_ : int
        Number of training samples.
    n_features_ : int
        Number of features in the training data.

    Notes
    -----
    **Complexity:**

    - Construction: :math:`O(1)` (data is simply stored)
    - Query: :math:`O(n \\cdot m)` per query point, where :math:`n` = samples, :math:`m` = features
    - Space: :math:`O(n \\cdot m)` for the stored training data

    **When to use LinearNNSearch:**

    - Small datasets where tree construction overhead is not justified
    - High-dimensional data where tree-based pruning provides little benefit
    - As a correctness baseline to verify tree-based search results
    - One-off queries where amortizing construction cost is not possible

    References
    ----------
    .. [Knuth1973] Knuth, D.E. (1973).
           **The Art of Computer Programming, Volume 3: Sorting and Searching.**
           *Addison-Wesley*.

    See Also
    --------
    :class:`~tuiml.algorithms.neighbors.search.KDTree` : Axis-aligned tree, faster in low dimensions.
    :class:`~tuiml.algorithms.neighbors.search.BallTree` : Spherical partitions, better in moderate to high dimensions.

    Examples
    --------
    Build a brute-force search and find the nearest neighbor:

    >>> from tuiml.algorithms.neighbors.search import LinearNNSearch
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> search = LinearNNSearch()
    >>> search.build(X)
    LinearNNSearch(n_samples=3)
    >>> dists, indices = search.query([3.1, 4.1], k=1)
    """

    def __init__(self):
        """Initialize LinearNNSearch."""
        super().__init__()

    def build(self, X: np.ndarray) -> "LinearNNSearch":
        """Store the training data for brute force search.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training data.

        Returns
        -------
        self : LinearNNSearch
            Returns the instance itself.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.X_ = X
        self._X_cpp = np.ascontiguousarray(X, dtype=np.float64)
        self.n_samples_, self.n_features_ = X.shape
        self._is_built = True

        return self

    def query(self, x: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Find the k nearest neighbors for a query point.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The query point.
        k : int, default=1
            Number of neighbors to find.

        Returns
        -------
        distances : np.ndarray of shape (k,)
            Distances to the k nearest neighbors.
        indices : np.ndarray of shape (k,)
            Indices of the k nearest neighbors in the training data.
        """
        self._check_is_built()
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(1, -1))
        dists, idxs = _cpp_nn.brute_knn_query(self._X_cpp, x, min(k, self.n_samples_))
        return dists[0], idxs[0]

    def query_batch(self, X: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Find the k nearest neighbors for multiple query points.

        Parameters
        ----------
        X : np.ndarray of shape (n_queries, n_features)
            The query points.
        k : int, default=1
            Number of neighbors to find.

        Returns
        -------
        distances : np.ndarray of shape (n_queries, k)
            Distances to the k nearest neighbors.
        indices : np.ndarray of shape (n_queries, k)
            Indices of the k nearest neighbors in the training data.
        """
        self._check_is_built()
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return _cpp_nn.brute_knn_query(self._X_cpp, X, min(k, self.n_samples_))

    def query_radius(self, x: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Find all neighbors within a specified radius.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            The query point.
        radius : float
            The maximum distance to search within.

        Returns
        -------
        distances : np.ndarray
            Distances to all neighbors within the radius.
        indices : np.ndarray
            Indices of neighbors within the radius.
        """
        self._check_is_built()
        x = np.asarray(x, dtype=float)

        # Compute all distances
        x_sq = np.sum(x ** 2)
        X_sq = np.sum(self.X_ ** 2, axis=1)
        distances_sq = x_sq + X_sq - 2 * self.X_ @ x
        distances_sq = np.maximum(distances_sq, 0)
        distances = np.sqrt(distances_sq)

        # Filter by radius
        mask = distances <= radius
        indices = np.where(mask)[0]

        # Sort by distance
        sorted_order = np.argsort(distances[indices])
        indices = indices[sorted_order]

        return distances[indices], indices

    def __repr__(self) -> str:
        """String representation."""
        if self._is_built:
            return f"LinearNNSearch(n_samples={self.n_samples_})"
        return "LinearNNSearch(not built)"
