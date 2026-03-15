"""Density-Based Spatial Clustering of Applications with Noise."""

import numpy as np
from typing import Dict, List, Any, Optional, Set

from tuiml.base.algorithms import Clusterer, clusterer
from tuiml.algorithms.clustering.distance import pairwise_distances

@clusterer(tags=["density-based", "noise-detection"], version="1.0.0")
class DBSCANClusterer(Clusterer):
    r"""
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Groups together points that are **closely packed** (points with many
    nearby neighbors), marking points in **low-density regions** as outliers.
    Unlike partitional methods, DBSCAN does not require the number of clusters
    to be specified in advance.

    Overview
    --------
    The algorithm discovers clusters through density-connected regions:

    1. Compute the :math:`\epsilon`-neighborhood for every point
    2. Identify **core points** that have at least ``min_samples`` neighbors
    3. Form clusters by connecting core points that are within :math:`\epsilon` of each other
    4. Assign **border points** to the cluster of a reachable core point
    5. Label remaining points as **noise** (label -1)

    Theory
    ------
    The :math:`\epsilon`-neighborhood of a point :math:`p` is defined as:

    .. math::
        N_{\epsilon}(p) = \{q \in D \mid \\text{dist}(p, q) \leq \epsilon\}

    A point :math:`p` is a **core point** if :math:`|N_{\epsilon}(p)| \geq \\text{min\_samples}`.

    A point :math:`q` is **directly density-reachable** from :math:`p` if:

    .. math::
        q \in N_{\epsilon}(p) \quad \\text{and} \quad |N_{\epsilon}(p)| \geq \\text{min\_samples}

    A cluster is a maximal set of **density-connected** points, where
    density-connectivity is the transitive closure of direct density-reachability.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be
        considered as a core point. This includes the point itself.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        The metric to use when calculating distance between instances
        in a feature array.

    Attributes
    ----------
    labels_ : np.ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.
    core_sample_indices_ : np.ndarray of shape (n_core_samples,)
        Indices of core samples.
    components_ : np.ndarray of shape (n_core_samples, n_features)
        Copy of each core sample found by training.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n^2)` time in the worst case. With a spatial index,
      this can be reduced to :math:`O(n \log n)`.
    - Space: :math:`O(n^2)` to store the distance matrix.

    **When to use DBSCANClusterer:**

    - When the number of clusters is unknown
    - Data contains arbitrarily shaped clusters (non-globular)
    - Outlier or noise detection is important
    - When clusters have similar densities

    References
    ----------
    .. [Ester1996] Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996).
           **A density-based algorithm for discovering clusters in
           large spatial databases with noise.**
           *KDD*, 96(34), pp. 226-231.

    .. [Schubert2017] Schubert, E., Sander, J., Ester, M., Kriegel, H.P., & Xu, X. (2017).
           **DBSCAN revisited, revisited: why and how you should (still) use DBSCAN.**
           *ACM Transactions on Database Systems*, 42(3), Article 19.
           DOI: `10.1145/3068335 <https://doi.org/10.1145/3068335>`_

    See Also
    --------
    :class:`~tuiml.algorithms.clustering.CanopyClusterer` : Fast approximate clustering using distance thresholds.
    :class:`~tuiml.algorithms.clustering.AgglomerativeClusterer` : Hierarchical clustering without noise detection.

    Examples
    --------
    Density-based clustering with noise detection:

    >>> import numpy as np
    >>> from tuiml.algorithms.clustering import DBSCANClusterer
    >>> X = np.array([[1, 2], [2, 2], [2, 3],
    ...               [8, 7], [8, 8], [25, 80]])
    >>> dbscan = DBSCANClusterer(eps=3, min_samples=2)
    >>> dbscan.fit(X)
    >>> dbscan.labels_
    array([ 0,  0,  0,  1,  1, -1])
    """

    def __init__(self, eps: float = 0.5,
                 min_samples: int = 5,
                 metric: str = 'euclidean'):
        """Initialize DBSCANClusterer with neighborhood parameters.

        Parameters
        ----------
        eps : float, default=0.5
            Neighborhood radius.
        min_samples : int, default=5
            Minimum samples for core point.
        metric : str, default='euclidean'
            Distance metric.
        """
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.core_sample_indices_ = None
        self.components_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "eps": {
                "type": "number",
                "default": 0.5,
                "minimum": 0,
                "description": "Neighborhood radius"
            },
            "min_samples": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "description": "Minimum samples for core point"
            },
            "metric": {
                "type": "string",
                "default": "euclidean",
                "enum": ["euclidean", "manhattan"],
                "description": "Distance metric"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "noise_detection"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n^2) time (naive), O(n * log(n)) with spatial index"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Ester, M., Kriegel, H.P., Sander, J., & Xu, X. (1996). "
            "A density-based algorithm for discovering clusters in large "
            "spatial databases with noise. KDD '96, 226-231."
        ]

    def _region_query(self, X: np.ndarray, point_idx: int,
                      dist_matrix: np.ndarray) -> np.ndarray:
        """Find all points within eps distance of a point.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Full dataset.
        point_idx : int
            Index of the query point.
        dist_matrix : np.ndarray of shape (n_samples, n_samples)
            Precomputed pairwise distance matrix.

        Returns
        -------
        neighbors : np.ndarray
            Indices of all points within ``eps`` distance of the query point.
        """
        return np.where(dist_matrix[point_idx] <= self.eps)[0]

    def _expand_cluster(self, X: np.ndarray, labels: np.ndarray,
                        point_idx: int, neighbors: np.ndarray,
                        cluster_id: int, dist_matrix: np.ndarray,
                        core_samples: Set[int]) -> None:
        """Expand a cluster from a core point using BFS.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Full dataset.
        labels : np.ndarray of shape (n_samples,)
            Mutable label array (modified in-place).
        point_idx : int
            Index of the seed core point.
        neighbors : np.ndarray
            Initial neighbor indices of the seed point.
        cluster_id : int
            Cluster label to assign.
        dist_matrix : np.ndarray of shape (n_samples, n_samples)
            Precomputed pairwise distance matrix.
        core_samples : set of int
            Mutable set of core sample indices (updated in-place).
        """
        labels[point_idx] = cluster_id

        # Use a queue for BFS expansion
        queue = list(neighbors)
        i = 0

        while i < len(queue):
            neighbor_idx = queue[i]
            i += 1

            if labels[neighbor_idx] == -1:
                # Was noise, now border point
                labels[neighbor_idx] = cluster_id

            elif labels[neighbor_idx] == -2:
                # Unvisited
                labels[neighbor_idx] = cluster_id

                # Check if this is a core point
                neighbor_neighbors = self._region_query(X, neighbor_idx, dist_matrix)

                if len(neighbor_neighbors) >= self.min_samples:
                    core_samples.add(neighbor_idx)
                    # Add new neighbors to queue
                    for nn in neighbor_neighbors:
                        if nn not in queue and labels[nn] < 0:
                            queue.append(nn)

    def fit(self, X: np.ndarray) -> "DBSCANClusterer":
        """Perform DBSCANClusterer clustering from features.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data to cluster.

        Returns
        -------
        self : DBSCANClusterer
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]

        # Compute pairwise distances
        dist_matrix = pairwise_distances(X, metric=self.metric)

        # Initialize labels: -2 = unvisited, -1 = noise
        labels = np.full(n_samples, -2, dtype=int)
        core_samples = set()

        cluster_id = 0

        for point_idx in range(n_samples):
            if labels[point_idx] != -2:
                # Already processed
                continue

            # Find neighbors
            neighbors = self._region_query(X, point_idx, dist_matrix)

            if len(neighbors) < self.min_samples:
                # Mark as noise (may later become border point)
                labels[point_idx] = -1
            else:
                # Core point: start a new cluster
                core_samples.add(point_idx)
                self._expand_cluster(
                    X, labels, point_idx, neighbors,
                    cluster_id, dist_matrix, core_samples
                )
                cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.array(sorted(core_samples))
        self.components_ = X[self.core_sample_indices_].copy()

        # Count clusters (excluding noise labeled as -1)
        unique_labels = set(labels)
        unique_labels.discard(-1)
        self.n_clusters_ = len(unique_labels)

        self._X = X  # Store for prediction
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data.

        New points are assigned to the nearest core point's cluster 
        if within `eps` distance, otherwise labeled as noise (-1).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to cluster.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels (noisy points are labeled -1).
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(self.core_sample_indices_) == 0:
            # No core samples, everything is noise
            return np.full(X.shape[0], -1, dtype=int)

        # Compute distances to core samples
        distances = pairwise_distances(X, self.components_, self.metric)

        # Assign to nearest core point's cluster if within eps
        labels = np.full(X.shape[0], -1, dtype=int)
        for i in range(X.shape[0]):
            min_dist_idx = np.argmin(distances[i])
            if distances[i, min_dist_idx] <= self.eps:
                core_idx = self.core_sample_indices_[min_dist_idx]
                labels[i] = self.labels_[core_idx]

        return labels

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels.

        Equivalent to `fit(X).labels_`.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X)
        return self.labels_

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            n_noise = np.sum(self.labels_ == -1)
            return (f"DBSCANClusterer(eps={self.eps}, min_samples={self.min_samples}, "
                   f"n_clusters={self.n_clusters_}, n_noise={n_noise})")
        return f"DBSCANClusterer(eps={self.eps}, min_samples={self.min_samples})"
