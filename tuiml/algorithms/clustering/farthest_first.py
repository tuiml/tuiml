"""Iteratively selects cluster centers farthest from existing ones."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Clusterer, clusterer
from tuiml.algorithms.clustering.distance import pairwise_distances

@clusterer(tags=["partitional", "initialization"], version="1.0.0")
class FarthestFirstClusterer(Clusterer):
    r"""
    Farthest-First Traversal clustering algorithm.

    A fast **greedy approximation** algorithm for the :math:`k`-center problem.
    It selects cluster centers by iteratively choosing the point that maximizes
    the minimum distance to any existing center, providing a **2-approximation**
    guarantee for the maximum cluster radius.

    Overview
    --------
    The algorithm builds a set of :math:`k` cluster centers greedily:

    1. Select the first center randomly from the data
    2. Compute the minimum distance from every point to the nearest existing center
    3. Choose the point with the **maximum** minimum distance as the next center
    4. Repeat steps 2--3 until :math:`k` centers are selected
    5. Assign each point to its nearest center

    Theory
    ------
    The center selection rule at step :math:`i` is:

    .. math::
        \mu_{i} = \\arg\max_{x \in X} \left( \min_{j < i} \\text{dist}(x, \mu_j) \\right)

    This greedy strategy guarantees that the maximum cluster radius is at most
    twice the optimal :math:`k`-center radius:

    .. math::
        \max_{x \in X} \min_{j} \\text{dist}(x, \mu_j) \leq 2 \cdot \\text{OPT}_k

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to find.
    random_state : int, optional, default=None
        Determines random number generation for the selection of the first
        cluster center.

    Attributes
    ----------
    cluster_centers_ : np.ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : np.ndarray of shape (n_samples,)
        Labels of each point (index of the nearest cluster center).

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \cdot k)` where :math:`n` is the number of samples
      and :math:`k` is the number of clusters.
    - Space: :math:`O(n + k)` to store labels and centers.

    **When to use FarthestFirstClusterer:**

    - As a fast initialization for K-Means or other iterative clusterers
    - When a quick, deterministic spread of cluster centers is needed
    - When you want worst-case guarantees on cluster radius
    - Very large datasets where K-Means++ is too slow

    References
    ----------
    .. [Hochbaum1985] Hochbaum, D. S., & Shmoys, D. B. (1985).
           **A best possible heuristic for the k-center problem.**
           *Mathematics of Operations Research*, 10(2), pp. 180-184.

    .. [Dasgupta2002] Dasgupta, S. (2002).
           **Performance Guarantees for Hierarchical Clustering.**
           *Proceedings of the 15th Annual Conference on Computational
           Learning Theory (COLT)*, pp. 351-363.

    See Also
    --------
    :class:`~tuiml.algorithms.clustering.KMeansClusterer` : Iterative centroid-based clustering often initialized with farthest-first.
    :class:`~tuiml.algorithms.clustering.CanopyClusterer` : Fast approximate clustering using distance thresholds.

    Examples
    --------
    Fast cluster center selection:

    >>> import numpy as np
    >>> from tuiml.algorithms.clustering import FarthestFirstClusterer
    >>> X = np.array([[1, 2], [10, 10], [1, 3], [11, 11]])
    >>> ff = FarthestFirstClusterer(n_clusters=2)
    >>> ff.fit(X)
    >>> ff.labels_
    array([0, 1, 0, 1])
    >>> ff.cluster_centers_
    array([[ 1.,  2.], [11., 11.]])
    """

    def __init__(self, n_clusters: int = 2,
                 random_state: Optional[int] = None):
        """Initialize FarthestFirstClusterer with clustering parameters.

        Parameters
        ----------
        n_clusters : int, default=2
            Number of clusters.
        random_state : int, optional
            Random seed.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_clusters": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "Number of clusters"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * k) time, O(n + k) space"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Hochbaum, D.S. & Shmoys, D.B. (1985). A best possible heuristic "
            "for the k-center problem. Math. Oper. Res., 10(2), 180-184.",
            "Dasgupta, S. (2002). Performance Guarantees for Hierarchical "
            "Clustering. COLT '02, 351-363."
        ]

    def fit(self, X: np.ndarray) -> "FarthestFirstClusterer":
        """Fit the FarthestFirstClusterer model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data to cluster.

        Returns
        -------
        self : FarthestFirstClusterer
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        # Handle case where n_clusters > n_samples
        k = min(self.n_clusters, n_samples)

        # Initialize cluster centers
        centers = np.zeros((k, n_features))
        center_indices = []

        # Choose first center randomly
        first_idx = rng.integers(n_samples)
        centers[0] = X[first_idx]
        center_indices.append(first_idx)

        # Track minimum distance to any center for each point
        min_distances = np.full(n_samples, np.inf)

        # Update distances for first center
        distances_to_first = np.sqrt(np.sum((X - centers[0]) ** 2, axis=1))
        min_distances = np.minimum(min_distances, distances_to_first)

        # Select remaining centers
        for i in range(1, k):
            # Choose point with maximum minimum distance
            next_idx = np.argmax(min_distances)
            centers[i] = X[next_idx]
            center_indices.append(next_idx)

            # Update minimum distances
            distances_to_new = np.sqrt(np.sum((X - centers[i]) ** 2, axis=1))
            min_distances = np.minimum(min_distances, distances_to_new)

        self.cluster_centers_ = centers
        self.center_indices_ = np.array(center_indices)

        # Assign labels
        distances = pairwise_distances(X, self.cluster_centers_)
        self.labels_ = np.argmin(distances, axis=1)
        self.n_clusters_ = k
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to cluster.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        distances = pairwise_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return f"FarthestFirstClusterer(n_clusters={self.n_clusters_})"
        return f"FarthestFirstClusterer(n_clusters={self.n_clusters})"
