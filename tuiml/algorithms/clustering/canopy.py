"""Fast, single-pass clustering using overlapping thresholds."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Clusterer, UpdateableClusterer, clusterer
from tuiml.algorithms.clustering.distance import pairwise_distances

@clusterer(tags=["fast", "approximate", "single-pass"], version="1.0.0")
class CanopyClusterer(UpdateableClusterer):
    r"""
    Canopy Clustering algorithm.

    A fast, **single-pass** clustering algorithm that groups data into overlapping
    subsets (canopies) using two distance thresholds, :math:`T_1` (loose) and
    :math:`T_2` (tight). It is often used as a **preprocessing step** before
    applying more expensive clustering algorithms like K-Means.

    Overview
    --------
    The algorithm performs a single pass over the data:

    1. Shuffle the data randomly
    2. Pick the first available point and create a new canopy with it as center
    3. Add any point within distance :math:`T_1` to the canopy
    4. Remove any point within distance :math:`T_2` from the candidate list
    5. Repeat until all points are processed or the maximum canopy count is reached

    Note that :math:`T_2 < T_1` is required for proper overlapping behavior.

    Theory
    ------
    Canopy clustering partitions data using two concentric distance thresholds
    around each canopy center :math:`c`:

    .. math::
        \\text{member}(x, c) = \\begin{cases} 1 & \\text{if } d(x, c) \leq T_1 \\ 0 & \\text{otherwise} \end{cases}

    .. math::
        \\text{remove}(x, c) = \\begin{cases} 1 & \\text{if } d(x, c) \leq T_2 \\ 0 & \\text{otherwise} \end{cases}

    The loose threshold :math:`T_1` determines canopy membership (allowing overlap),
    while the tight threshold :math:`T_2` prevents a point from becoming a future
    canopy center, ensuring separation between canopy centers.

    Parameters
    ----------
    n_clusters : int, default=-1
        Target number of canopies to keep. If set to -1, all discovered
        canopies are returned.
    t1 : float, optional, default=None
        Loose distance threshold. If None, it is estimated from data statistics.
    t2 : float, optional, default=None
        Tight distance threshold. If None, it is estimated from data statistics.
    max_canopies : int, default=-1
        Maximum number of canopies to create. -1 for unlimited.
    random_state : int, optional, default=None
        Determines random number generation for shuffling the data.

    Attributes
    ----------
    canopy_centers_ : np.ndarray of shape (n_clusters, n_features)
        Coordinates of discovered canopy centers.
    labels_ : np.ndarray of shape (n_samples,)
        Cluster labels for each point (index of the nearest canopy center).
    t1_ : float
        The actual T1 threshold used.
    t2_ : float
        The actual T2 threshold used.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \cdot k)` where :math:`n` is the number of samples
      and :math:`k` is the number of resulting canopies.
    - Space: :math:`O(k \cdot m)` where :math:`m` is the number of features.

    **When to use CanopyClusterer:**

    - As a fast preprocessing step before K-Means or EM clustering
    - When you need approximate clustering with very large datasets
    - When overlapping cluster membership is acceptable
    - Streaming or single-pass clustering scenarios

    References
    ----------
    .. [McCallum2000] McCallum, A., Nigam, K., & Ungar, L. H. (2000).
           **Efficient clustering of high dimensional data sets with
           application to reference matching.**
           *Proceedings of the sixth ACM SIGKDD international conference
           on Knowledge discovery and data mining*, pp. 169-178.

    See Also
    --------
    :class:`~tuiml.algorithms.clustering.KMeansClusterer` : Centroid-based clustering often initialized with canopies.
    :class:`~tuiml.algorithms.clustering.GaussianMixtureClusterer` : Probabilistic clustering that benefits from canopy preprocessing.

    Examples
    --------
    Basic canopy clustering with explicit thresholds:

    >>> import numpy as np
    >>> from tuiml.algorithms.clustering import CanopyClusterer
    >>> X = np.array([[1, 2], [1.5, 1.8], [5, 5], [6, 4]])
    >>> canopy = CanopyClusterer(t1=2.0, t2=1.0)
    >>> canopy.fit(X)
    >>> canopy.n_clusters_
    2
    >>> canopy.predict([[1.2, 1.9]])
    array([0])
    """

    def __init__(self, n_clusters: int = -1,
                 t1: Optional[float] = None,
                 t2: Optional[float] = None,
                 max_canopies: int = -1,
                 random_state: Optional[int] = None):
        """Initialize CanopyClusterer with thresholds.

        Parameters
        ----------
        n_clusters : int, default=-1
            Target number of clusters (-1 for automatic).
        t1 : float, optional
            Loose distance threshold.
        t2 : float, optional
            Tight distance threshold.
        max_canopies : int, default=-1
            Maximum canopies to create.
        random_state : int, optional
            Random seed.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.t1 = t1
        self.t2 = t2
        self.max_canopies = max_canopies
        self.random_state = random_state
        self.canopy_centers_ = None
        self.t1_ = None
        self.t2_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_clusters": {
                "type": "integer",
                "default": -1,
                "description": "Target number of clusters (-1 for auto)"
            },
            "t1": {
                "type": "number",
                "default": None,
                "description": "Loose distance threshold"
            },
            "t2": {
                "type": "number",
                "default": None,
                "description": "Tight distance threshold"
            },
            "max_canopies": {
                "type": "integer",
                "default": -1,
                "description": "Maximum canopies (-1 for unlimited)"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "incremental"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * k) time, O(k) space"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "McCallum, A., Nigam, K., & Ungar, L.H. (2000). Efficient "
            "Clustering of High Dimensional Data Sets with Application "
            "to Reference Matching. KDD '00, 169-178."
        ]

    def _estimate_thresholds(self, X: np.ndarray) -> tuple:
        """Estimate T1 and T2 based on data statistics.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data used to compute threshold estimates.

        Returns
        -------
        t1 : float
            Estimated loose distance threshold.
        t2 : float
            Estimated tight distance threshold.
        """
        # Use standard deviations to estimate thresholds
        stds = np.std(X, axis=0)
        mean_std = np.mean(stds)

        # T2 should be tighter than T1
        t2 = mean_std * 0.5
        t1 = mean_std * 1.5

        return t1, t2

    def fit(self, X: np.ndarray) -> "CanopyClusterer":
        """Fit the CanopyClusterer clustering model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data to cluster.

        Returns
        -------
        self : CanopyClusterer
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        # Shuffle data for randomization
        indices = rng.permutation(n_samples)
        X_shuffled = X[indices]

        # Estimate thresholds if not provided
        if self.t1 is None or self.t2 is None:
            t1_est, t2_est = self._estimate_thresholds(X)
            self.t1_ = self.t1 if self.t1 is not None else t1_est
            self.t2_ = self.t2 if self.t2 is not None else t2_est
        else:
            self.t1_ = self.t1
            self.t2_ = self.t2

        # Ensure t2 <= t1
        if self.t2_ > self.t1_:
            self.t2_ = self.t1_

        # Track which points can still become canopy centers
        available = np.ones(n_samples, dtype=bool)
        canopy_centers = []
        canopy_members = []  # List of member indices for each canopy

        for i in range(n_samples):
            if not available[i]:
                continue

            if self.max_canopies > 0 and len(canopy_centers) >= self.max_canopies:
                break

            # Create new canopy
            center = X_shuffled[i]
            canopy_centers.append(center)
            members = [indices[i]]  # Original indices

            # Find all points within T1 (members) and mark T2 as unavailable
            for j in range(i + 1, n_samples):
                if not available[j]:
                    continue

                dist = np.sqrt(np.sum((X_shuffled[j] - center) ** 2))

                if dist <= self.t1_:
                    members.append(indices[j])

                if dist <= self.t2_:
                    available[j] = False

            canopy_members.append(members)

        self.canopy_centers_ = np.array(canopy_centers)
        self.canopy_members_ = canopy_members
        self.cluster_centers_ = self.canopy_centers_

        # If specific number of clusters requested, select top canopies
        if self.n_clusters > 0 and len(canopy_centers) > self.n_clusters:
            # Select canopies with most members
            sizes = [len(m) for m in canopy_members]
            top_indices = np.argsort(sizes)[-self.n_clusters:]
            self.canopy_centers_ = self.canopy_centers_[top_indices]
            self.canopy_members_ = [canopy_members[i] for i in top_indices]
            self.cluster_centers_ = self.canopy_centers_

        # Assign labels to all training points
        self.labels_ = self._assign_labels(X)
        self.n_clusters_ = len(self.canopy_centers_)
        self._X = X
        self._is_fitted = True

        return self

    def _assign_labels(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to nearest canopy center.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data points to assign.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Index of the nearest canopy center for each point.
        """
        if len(self.canopy_centers_) == 0:
            return np.zeros(X.shape[0], dtype=int)

        distances = pairwise_distances(X, self.canopy_centers_)
        return np.argmin(distances, axis=1)

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

        return self._assign_labels(X)

    def update(self, X: np.ndarray) -> "CanopyClusterer":
        """Update the model with new instances incrementally.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data to incorporate into the canopy model.

        Returns
        -------
        self : CanopyClusterer
            Updated estimator.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self._is_fitted:
            return self.fit(X)

        # Add points to existing canopies or create new ones
        for point in X:
            # Check if point is within T2 of any existing canopy
            within_t2 = False
            for center in self.canopy_centers_:
                dist = np.sqrt(np.sum((point - center) ** 2))
                if dist <= self.t2_:
                    within_t2 = True
                    break

            if not within_t2:
                # Create new canopy
                if self.max_canopies <= 0 or len(self.canopy_centers_) < self.max_canopies:
                    self.canopy_centers_ = np.vstack([self.canopy_centers_, point])
                    self.cluster_centers_ = self.canopy_centers_
                    self.n_clusters_ = len(self.canopy_centers_)

        return self

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"CanopyClusterer(n_clusters={self.n_clusters_}, "
                   f"t1={self.t1_:.3f}, t2={self.t2_:.3f})")
        return f"CanopyClusterer(n_clusters={self.n_clusters})"
