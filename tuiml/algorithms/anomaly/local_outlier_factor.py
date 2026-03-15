"""Local Outlier Factor (LOF) anomaly detection algorithm."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List
from tuiml.base.algorithms import Classifier, classifier

@classifier(tags=["anomaly-detection", "density-based", "neighbors"], version="1.0.0")
class LocalOutlierFactorDetector(Classifier):
    """Local Outlier Factor for density-based anomaly detection.

    LOF detects anomalies by comparing the **local density** of a point with the
    densities of its neighbors. Points in significantly sparser regions than their
    neighbors are identified as outliers. Unlike global density methods, LOF adapts
    to **local density variations** in the data.

    Overview
    --------
    The algorithm works through four main steps:

    1. Find the k-nearest neighbors for each point
    2. Compute the **reachability distance** (smoothed distance to neighbors)
    3. Calculate **Local Reachability Density (LRD)** for each point
    4. Compute the **LOF score** as the ratio of neighbor densities to point density

    The key insight: Outliers have much lower density than their neighbors,
    resulting in LOF scores significantly greater than 1.

    Theory
    ------
    For a point :math:`A` with k-nearest neighbors :math:`N_k(A)`:

    **1. k-distance:** Distance to the k-th nearest neighbor

    **2. Reachability distance:**

    .. math::
        \\text{reach-dist}_k(A, B) = \\max(k\\text{-distance}(B), d(A, B))

    This "smooths" distances to prevent instability when points are very close.

    **3. Local Reachability Density (LRD):**

    .. math::
        \\text{LRD}_k(A) = \\frac{1}{\\frac{1}{|N_k(A)|} \\sum_{B \\in N_k(A)} \\text{reach-dist}_k(A, B)}

    Higher LRD → point is in a denser region.

    **4. Local Outlier Factor:**

    .. math::
        \\text{LOF}_k(A) = \\frac{1}{|N_k(A)|} \\sum_{B \\in N_k(A)} \\frac{\\text{LRD}_k(B)}{\\text{LRD}_k(A)}

    **Score interpretation:**

    - :math:`\\text{LOF} \\approx 1` → Normal point (similar density to neighbors)
    - :math:`\\text{LOF} < 1` → Denser than neighbors (inlier)
    - :math:`\\text{LOF} > 1` → Sparser than neighbors (outlier)
    - :math:`\\text{LOF} \\gg 1` → Strong outlier

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors to use for local density estimation. Larger values
        consider more global structure; smaller values are more sensitive to
        local variations.

    contamination : float, default=0.1
        Expected proportion of outliers in the dataset. Must be in the range
        ``(0, 0.5]``. Used to set the decision threshold.

    metric : str, default="euclidean"
        Distance metric for computing neighbor distances:

        - ``"euclidean"`` — Euclidean (L2) distance
        - ``"manhattan"`` — Manhattan (L1) distance
        - ``"chebyshev"`` — Chebyshev (L∞) distance

    novelty : bool, default=False
        Detection mode:

        - ``False`` — Outlier detection (fit and predict on same data)
        - ``True`` — Novelty detection (predict on new, unseen data)

    Attributes
    ----------
    X_train_ : np.ndarray
        Training data (stored for computing densities of new points).

    neighbors_indices_ : np.ndarray of shape (n_samples, n_neighbors)
        Indices of k-nearest neighbors for each training sample.

    neighbors_distances_ : np.ndarray of shape (n_samples, n_neighbors)
        Distances to k-nearest neighbors for each training sample.

    lrd_ : np.ndarray of shape (n_samples,)
        Local Reachability Density for each training sample.

    lof_scores_ : np.ndarray of shape (n_samples,)
        LOF scores for each training sample.

    threshold_ : float
        Decision threshold separating normal from anomalous instances.
        Computed based on the contamination parameter.

    n_features_in_ : int
        Number of features observed during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n^2 \\log n)` where :math:`n` = number of samples
    - Prediction: :math:`O(n \\cdot k)` where :math:`k` = n_neighbors

    **When to use LOF:**

    - Data with varying local densities (clusters of different densities)
    - When global outlier detection is insufficient
    - Medium-sized datasets (becomes slow for very large datasets)
    - When you want interpretable density-based scores

    **Limitations:**

    - Quadratic complexity limits scalability to large datasets
    - Sensitive to choice of ``n_neighbors`` parameter
    - Requires storing training data for novelty detection
    - Performance degrades in very high dimensions (curse of dimensionality)

    References
    ----------
    .. [Breunig2000] Breunig, M.M., Kriegel, H.P., Ng, R.T. and Sander, J. (2000).
           **LOF: identifying density-based local outliers.**
           *ACM Sigmod Record*, 29(2), pp. 93-104.
           DOI: `10.1145/335191.335388 <https://doi.org/10.1145/335191.335388>`_

    See Also
    --------
    :class:`~tuiml.algorithms.anomaly.IsolationForest` : Tree-based ensemble anomaly detection.
    :class:`~tuiml.algorithms.anomaly.ABOD` : Angle-based outlier detection for high-dimensional data.

    Examples
    --------
    Basic usage for outlier detection:

    >>> from tuiml.algorithms.anomaly import LocalOutlierFactorDetector
    >>> import numpy as np
    >>>
    >>> # Create data with one clear outlier
    >>> X = np.array([[1, 2], [2, 3], [3, 3], [2, 2], [10, 10]])
    >>>
    >>> # Fit and predict on training data
    >>> clf = LocalOutlierFactorDetector(n_neighbors=3, contamination=0.2)
    >>> predictions = clf.fit_predict(X)
    >>> print(predictions)
    [ 1  1  1  1 -1]

    Novelty detection on new data:

    >>> # Train on normal data
    >>> X_train = np.array([[1, 2], [2, 3], [3, 3], [2, 2]])
    >>> X_test = np.array([[2.5, 2.5], [10, 10]])
    >>>
    >>> # Use novelty mode to predict on new data
    >>> clf = LocalOutlierFactorDetector(n_neighbors=3, novelty=True)
    >>> clf.fit(X_train)
    >>> predictions = clf.predict(X_test)
    >>> print(predictions)
    [ 1 -1]
    >>>
    >>> # Get LOF scores
    >>> scores = clf.decision_function(X_test)
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        metric: str = "euclidean",
        novelty: bool = False,
    ):
        """Initialize Local Outlier Factor.

        Parameters
        ----------
        n_neighbors : int, default=20
            Number of neighbors for density estimation.
        contamination : float, default=0.1
            Expected proportion of outliers.
        metric : str, default="euclidean"
            Distance metric.
        novelty : bool, default=False
            Enable novelty detection mode.
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.novelty = novelty

        # Fitted attributes
        self.X_train_ = None
        self.neighbors_indices_ = None
        self.neighbors_distances_ = None
        self.lrd_ = None
        self.lof_scores_ = None
        self.threshold_ = None
        self.n_features_in_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "n_neighbors": {
                "type": "integer",
                "default": 20,
                "minimum": 1,
                "description": "Number of neighbors for density estimation"
            },
            "contamination": {
                "type": "number",
                "default": 0.1,
                "minimum": 0.0,
                "maximum": 0.5,
                "description": "Expected proportion of outliers"
            },
            "metric": {
                "type": "string",
                "default": "euclidean",
                "enum": ["euclidean", "manhattan", "chebyshev"],
                "description": "Distance metric to use"
            },
            "novelty": {
                "type": "boolean",
                "default": False,
                "description": "Enable novelty detection mode"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return [
            "numeric",
            "binary_class",
            "unsupervised",
            "anomaly_detection",
            "novelty_detection"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(n² log n), Prediction: O(n * k), where n=samples, k=n_neighbors"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Breunig et al., 2000. LOF: identifying density-based local outliers. ACM Sigmod Record."
        ]

    def _compute_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between X and Y.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples_X, n_features)
            First array.
        Y : np.ndarray of shape (n_samples_Y, n_features)
            Second array.

        Returns
        -------
        distances : np.ndarray of shape (n_samples_X, n_samples_Y)
            Pairwise distances.
        """
        if self.metric == "euclidean":
            # Broadcasting: (n, 1, d) - (1, m, d) -> (n, m, d)
            diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
            return np.sqrt(np.sum(diff ** 2, axis=2))
        elif self.metric == "manhattan":
            diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
            return np.sum(np.abs(diff), axis=2)
        elif self.metric == "chebyshev":
            diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
            return np.max(np.abs(diff), axis=2)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _find_k_neighbors(self, X: np.ndarray, Y: np.ndarray, k: int):
        """Find k-nearest neighbors of X in Y.

        Parameters
        ----------
        X : np.ndarray
            Query points.
        Y : np.ndarray
            Reference points.
        k : int
            Number of neighbors.

        Returns
        -------
        indices : np.ndarray
            Indices of k-nearest neighbors.
        distances : np.ndarray
            Distances to k-nearest neighbors.
        """
        distances = self._compute_distance(X, Y)

        # For each row, find k smallest distances
        # np.argpartition is faster than full sort
        k_smallest_idx = np.argpartition(distances, k - 1, axis=1)[:, :k]

        # Get actual distances
        k_distances = np.take_along_axis(distances, k_smallest_idx, axis=1)

        # Sort within the k neighbors
        sorted_idx = np.argsort(k_distances, axis=1)
        k_smallest_idx = np.take_along_axis(k_smallest_idx, sorted_idx, axis=1)
        k_distances = np.take_along_axis(k_distances, sorted_idx, axis=1)

        return k_smallest_idx, k_distances

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LocalOutlierFactorDetector":
        """Fit the LOF model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray or None, default=None
            Ignored. Present for API consistency.

        Returns
        -------
        self : LocalOutlierFactorDetector
            Fitted estimator.
        """
        X = np.atleast_2d(X)
        n_samples = len(X)
        self.n_features_in_ = X.shape[1]

        if self.n_neighbors >= n_samples:
            raise ValueError(f"n_neighbors ({self.n_neighbors}) must be < n_samples ({n_samples})")

        self.X_train_ = X.copy()

        # Find k-nearest neighbors for each point
        # When finding neighbors in training set, we need k+1 (first is always the point itself)
        k = min(self.n_neighbors + 1, n_samples)
        neighbors_idx, neighbors_dist = self._find_k_neighbors(X, X, k)

        # Remove self (first neighbor with distance 0)
        self.neighbors_indices_ = neighbors_idx[:, 1:]
        self.neighbors_distances_ = neighbors_dist[:, 1:]

        # Compute k-distance (distance to k-th neighbor)
        k_distances = self.neighbors_distances_[:, -1]

        # Compute reachability distances
        # reach_dist(A, B) = max(k_dist(B), dist(A, B))
        reach_dists = np.maximum(
            k_distances[self.neighbors_indices_],  # k-distances of neighbors
            self.neighbors_distances_  # actual distances
        )

        # Compute Local Reachability Density (LRD)
        # LRD(A) = 1 / (mean(reach_dist(A, B)) for B in neighbors of A)
        mean_reach_dists = np.mean(reach_dists, axis=1)
        # Handle case where mean_reach_dist is 0 (all neighbors at same location)
        self.lrd_ = np.where(mean_reach_dists > 0, 1.0 / mean_reach_dists, np.inf)

        # Compute LOF scores
        # LOF(A) = mean(LRD(B)) / LRD(A) for B in neighbors of A
        neighbors_lrd = self.lrd_[self.neighbors_indices_]
        mean_neighbors_lrd = np.mean(neighbors_lrd, axis=1)
        self.lof_scores_ = np.where(
            np.isfinite(self.lrd_) & (self.lrd_ > 0),
            mean_neighbors_lrd / self.lrd_,
            1.0  # If LRD is inf or 0, LOF = 1 (normal)
        )

        # Determine threshold based on contamination
        self.threshold_ = np.percentile(self.lof_scores_, 100 * (1 - self.contamination))

        self._is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Shifted opposite of LOF scores. Higher values indicate anomalies.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Anomaly scores. Higher values indicate anomalies.
        """
        self._check_is_fitted()
        X = np.atleast_2d(X)

        if not self.novelty:
            # For outlier detection, only work on training data
            if not np.array_equal(X, self.X_train_):
                raise ValueError(
                    "decision_function is not available for new data when novelty=False. "
                    "Use fit_predict or set novelty=True."
                )
            scores = self.lof_scores_
        else:
            # For novelty detection, compute LOF for new samples
            scores = self._compute_lof_scores(X)

        # Return negative so higher = more anomalous
        return -(scores - 1.0)  # Shift so normal (LOF=1) gives score 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict if samples are anomalies or not.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            -1 for anomalies, 1 for normal instances.
        """
        self._check_is_fitted()
        X = np.atleast_2d(X)

        if not self.novelty:
            # For outlier detection
            if not np.array_equal(X, self.X_train_):
                raise ValueError(
                    "predict is not available for new data when novelty=False. "
                    "Use fit_predict or set novelty=True."
                )
            lof_scores = self.lof_scores_
        else:
            # For novelty detection
            lof_scores = self._compute_lof_scores(X)

        is_inlier = lof_scores <= self.threshold_
        return np.where(is_inlier, 1, -1)

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and predict anomalies in training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray or None, default=None
            Ignored.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            -1 for anomalies, 1 for normal instances.
        """
        self.fit(X, y)
        is_inlier = self.lof_scores_ <= self.threshold_
        return np.where(is_inlier, 1, -1)

    def _compute_lof_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute LOF scores for new samples (novelty detection).

        Parameters
        ----------
        X : np.ndarray
            New samples.

        Returns
        -------
        lof_scores : np.ndarray
            LOF scores.
        """
        # Find neighbors in training set
        neighbors_idx, neighbors_dist = self._find_k_neighbors(X, self.X_train_, self.n_neighbors)

        # Get k-distances of training neighbors
        k_distances_train = self.neighbors_distances_[neighbors_idx, -1]

        # Compute reachability distances
        reach_dists = np.maximum(k_distances_train, neighbors_dist)

        # Compute LRD for new samples
        mean_reach_dists = np.mean(reach_dists, axis=1)
        lrd_new = np.where(mean_reach_dists > 0, 1.0 / mean_reach_dists, np.inf)

        # Compute LOF scores
        neighbors_lrd = self.lrd_[neighbors_idx]
        mean_neighbors_lrd = np.mean(neighbors_lrd, axis=1)
        lof_scores = np.where(
            np.isfinite(lrd_new) & (lrd_new > 0),
            mean_neighbors_lrd / lrd_new,
            1.0
        )

        return lof_scores

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Alias for decision_function for compatibility.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Anomaly scores.
        """
        return self.decision_function(X)
