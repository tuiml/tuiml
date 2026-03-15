"""Angle-Based Outlier Detection (ABODDetector) algorithm."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List
from tuiml.base.algorithms import Classifier, classifier

@classifier(tags=["anomaly-detection", "angle-based", "geometric"], version="1.0.0")
class ABODDetector(Classifier):
    """Angle-Based Outlier Detection for high-dimensional anomaly detection.

    ABODDetector detects anomalies by analyzing the **variance of angles** formed between 
    a data point and all other point pairs. Unlike distance-based methods, ABODDetector is 
    particularly effective in **high-dimensional spaces** where distances become less meaningful.

    Overview
    --------
    The algorithm works by computing angle variance for each point:

    1. For each point A, select all pairs of other points (B, C)
    2. Compute the angle between vectors AB and AC
    3. Calculate the variance of these angles
    4. Points with **low angle variance** are anomalies

    The intuition: Normal points in dense regions see neighbors from many different 
    angles (high variance), while outliers in sparse regions see all points from 
    similar angles (low variance).

    Theory
    ------
    For each point :math:`A`, the Angle-Based Outlier Factor (ABOF) is:

    .. math::
        \\text{ABOF}(A) = \\text{Var}_{B,C} \\left( \\angle(\\vec{AB}, \\vec{AC}) \\right)

    The angle between vectors :math:`\\vec{AB}` and :math:`\\vec{AC}` is:

    .. math::
        \\cos(\\theta) = \\frac{\\vec{AB} \\cdot \\vec{AC}}{\\|\\vec{AB}\\| \\cdot \\|\\vec{AC}\\|}

    **Weighted ABOF (WABOF)** incorporates distance information:

    .. math::
        \\text{WABOF}(A) = \\frac{\\sum_{B,C} w(B,C) \\cdot \\angle(\\vec{AB}, \\vec{AC})^2}{\\sum_{B,C} w(B,C)}

    where:

    - :math:`w(B,C) = \\frac{1}{\\|\\vec{AB}\\| \\cdot \\|\\vec{AC}\\|}` — Weight inversely proportional to distances
    - Lower weight for distant point pairs (less influential)

    **Score interpretation:**

    - Low ABOF score → Anomaly (isolated, low angle variance)
    - High ABOF score → Normal point (dense region, high angle variance)

    Parameters
    ----------
    contamination : float, default=0.1
        Expected proportion of outliers in the dataset. Must be in the 
        range ``(0, 0.5]``. Used to set the decision threshold.

    n_neighbors : int or None, default=None
        Number of neighbors to use for angle computation:
        
        - ``None`` — Use all points (most accurate, slowest)
        - ``int`` — Use k nearest neighbors (faster approximation)

    method : str, default="fast"
        Computation method:
        
        - ``"full"`` — Compute angles for all point pairs  
        - ``"fast"`` — Use approximate method with neighbor sampling

    weighted : bool, default=True
        If ``True``, use weighted ABOF (distance-weighted angles).
        If ``False``, use simple angle variance.

    Attributes
    ----------
    abof_scores_ : np.ndarray
        ABOF scores for each training sample. Lower scores indicate outliers.

    threshold_ : float
        Decision threshold separating normal instances from anomalies.
        Computed based on the contamination parameter.

    n_features_in_ : int
        Number of features observed during ``fit()``.

    X_train_ : np.ndarray
        Training data (stored for prediction on new samples).

    Notes
    -----
    **Complexity:**

    - Full method: :math:`O(n^3)` where :math:`n` = number of samples
    - Fast method: :math:`O(n^2 \\cdot k)` where :math:`k` = n_neighbors

    **When to use ABODDetector:**

    - High-dimensional datasets (where distance-based methods fail)
    - When interpretability through geometric properties is desired
    - Small to medium datasets (computational cost grows cubically)
    - When you want rotation-invariant anomaly detection

    **Limitations:**

    - Computationally expensive for large datasets
    - Requires storing training data for prediction
    - May struggle with datasets having uniform angular distributions

    References
    ----------
    .. [Kriegel2008] Kriegel, H.P., Schubert, M. and Zimek, A. (2008).
           **Angle-based outlier detection in high-dimensional data.**
           *Proceedings of the 14th ACM SIGKDD International Conference on Knowledge 
           Discovery and Data Mining (KDD)*, pp. 444-452.
           DOI: `10.1145/1401890.1401946 <https://doi.org/10.1145/1401890.1401946>`_

    .. [Kriegel2009] Kriegel, H.P., Kröger, P., Schubert, E. and Zimek, A. (2009).
           **Loop: local outlier probabilities.**
           *Proceedings of the 18th ACM Conference on Information and Knowledge 
           Management (CIKM)*, pp. 1649-1652.
           DOI: `10.1145/1645953.1646195 <https://doi.org/10.1145/1645953.1646195>`_

    See Also
    --------
    :class:`~tuiml.algorithms.anomaly.IsolationForest` : Tree-based ensemble anomaly detection.
    :class:`~tuiml.algorithms.anomaly.LocalOutlierFactor` : Density-based anomaly detection.

    Examples
    --------
    Basic usage for detecting anomalies in 2D data:

    >>> from tuiml.algorithms.anomaly import ABODDetector
    >>> import numpy as np
    >>> 
    >>> # Create data with one clear outlier
    >>> X = np.array([[1, 2], [2, 3], [3, 3], [2, 2], [10, 10]])
    >>> 
    >>> # Fit the model
    >>> clf = ABODDetector(contamination=0.2, n_neighbors=3, weighted=True)
    >>> clf.fit(X)
    >>> 
    >>> # Predict: -1 for anomalies, 1 for normal points
    >>> predictions = clf.predict(X)
    >>> print(predictions)
    [ 1  1  1  1 -1]
    >>> 
    >>> # Get ABODDetector scores (lower = more anomalous)
    >>> scores = clf.decision_function(X)
    
    High-dimensional example:
    
    >>> # Generate high-dimensional data
    >>> np.random.seed(42)
    >>> X_train = np.random.randn(100, 20)  # 100 samples, 20 features
    >>> X_train[0] *= 3  # Make first point an outlier
    >>> 
    >>> # Use fast method with limited neighbors for efficiency
    >>> clf = ABODDetector(contamination=0.05, n_neighbors=10, method="fast")
    >>> clf.fit(X_train)
    >>> anomalies = clf.predict(X_train)
    >>> print(f"Detected {(anomalies == -1).sum()} anomalies")
    Detected 5 anomalies
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_neighbors: int | None = None,
        method: str = "fast",
        weighted: bool = True,
    ):
        """Initialize ABODDetector.

        Parameters
        ----------
        contamination : float, default=0.1
            Expected proportion of outliers.
        n_neighbors : int or None, default=None
            Number of neighbors for computation.
        method : str, default="fast"
            Computation method ("full" or "fast").
        weighted : bool, default=True
            Use weighted ABOF.
        """
        super().__init__()
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.method = method
        self.weighted = weighted

        # Fitted attributes
        self.abof_scores_ = None
        self.threshold_ = None
        self.n_features_in_ = None
        self.X_train_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "contamination": {
                "type": "number",
                "default": 0.1,
                "minimum": 0.0,
                "maximum": 0.5,
                "description": "Expected proportion of outliers"
            },
            "n_neighbors": {
                "type": ["integer", "null"],
                "default": None,
                "minimum": 2,
                "description": "Number of neighbors for angle computation"
            },
            "method": {
                "type": "string",
                "default": "fast",
                "enum": ["full", "fast"],
                "description": "Computation method"
            },
            "weighted": {
                "type": "boolean",
                "default": True,
                "description": "Use weighted ABOF"
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
            "high_dimensional"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Full: O(n³), Fast: O(n² * k), where n=samples, k=neighbors"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Kriegel et al., 2008. Angle-based outlier detection in high-dimensional data. KDD.",
            "Kriegel et al., 2009. Loop: local outlier probabilities. CIKM."
        ]

    def _compute_angle_variance(
        self,
        point_idx: int,
        X: np.ndarray,
        neighbor_indices: Optional[np.ndarray] = None
    ) -> float:
        """Compute angle variance for a single point.

        Parameters
        ----------
        point_idx : int
            Index of the point.
        X : np.ndarray
            Data matrix.
        neighbor_indices : np.ndarray or None
            Indices of neighbors to use. If None, use all points.

        Returns
        -------
        abof : float
            ABOF score (angle variance or weighted angle variance).
        """
        point_a = X[point_idx]

        if neighbor_indices is None:
            # Use all other points
            indices = np.arange(len(X))
            indices = indices[indices != point_idx]
        else:
            indices = neighbor_indices[neighbor_indices != point_idx]

        if len(indices) < 2:
            return 0.0  # Cannot compute angles with < 2 neighbors

        # Compute difference vectors
        diff_vectors = X[indices] - point_a

        # Compute all pairwise angles
        angles = []
        weights = []

        n_neighbors = len(indices)
        for i in range(n_neighbors):
            for j in range(i + 1, n_neighbors):
                vec_ab = diff_vectors[i]
                vec_ac = diff_vectors[j]

                # Compute angle using cosine similarity
                norm_ab = np.linalg.norm(vec_ab)
                norm_ac = np.linalg.norm(vec_ac)

                if norm_ab > 1e-10 and norm_ac > 1e-10:
                    cos_angle = np.dot(vec_ab, vec_ac) / (norm_ab * norm_ac)
                    # Clip to valid range for numerical stability
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)

                    angles.append(angle)

                    if self.weighted:
                        # Weight by inverse of distance product
                        weight = 1.0 / (norm_ab * norm_ac)
                        weights.append(weight)

        if len(angles) == 0:
            return 0.0

        angles = np.array(angles)

        if self.weighted and len(weights) > 0:
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize
            # Weighted variance
            mean_angle = np.sum(weights * angles)
            variance = np.sum(weights * (angles - mean_angle) ** 2)
        else:
            # Simple variance
            variance = np.var(angles)

        return variance

    def fit(self, X: np.ndarray, _y: Optional[np.ndarray] = None) -> "ABODDetector":
        """Fit the ABODDetector model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        _y : np.ndarray or None, default=None
            Ignored. Present for API consistency.

        Returns
        -------
        self : ABODDetector
            Fitted estimator.
        """
        X = np.atleast_2d(X)
        n_samples = len(X)
        self.n_features_in_ = X.shape[1]
        self.X_train_ = X.copy()

        # Determine neighbors to use
        if self.n_neighbors is None:
            k = n_samples
        else:
            k = min(self.n_neighbors, n_samples)

        # Compute ABOF scores for all points
        self.abof_scores_ = np.zeros(n_samples)

        for i in range(n_samples):
            if self.method == "fast" and self.n_neighbors is not None:
                # Find k nearest neighbors
                distances = np.linalg.norm(X - X[i], axis=1)
                neighbor_idx = np.argsort(distances)[:k]
            else:
                neighbor_idx = None

            self.abof_scores_[i] = self._compute_angle_variance(i, X, neighbor_idx)

        # Determine threshold
        # Lower ABOF = outlier, so use lower percentile
        self.threshold_ = np.percentile(self.abof_scores_, 100 * self.contamination)

        self._is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Returns ABOF scores. Lower scores indicate anomalies.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            ABOF scores. Lower scores indicate anomalies.
        """
        self._check_is_fitted()
        X = np.atleast_2d(X)

        # For new data, compute ABOF using training data as reference
        scores = np.zeros(len(X))

        for i, x in enumerate(X):
            # Use training data to compute angles
            # Treat x as the point A, and training data as neighbors
            X_extended = np.vstack([self.X_train_, x[np.newaxis, :]])
            point_idx = len(X_extended) - 1

            if self.n_neighbors is not None:
                # Find k nearest neighbors from training data
                distances = np.linalg.norm(self.X_train_ - x, axis=1)
                neighbor_idx = np.argsort(distances)[:self.n_neighbors]
            else:
                neighbor_idx = None

            scores[i] = self._compute_angle_variance(point_idx, X_extended, neighbor_idx)

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict if samples are inliers or outliers.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            -1 for anomalies (outliers), 1 for normal (inliers).
        """
        self._check_is_fitted()
        X = np.atleast_2d(X)

        # Check if predicting on training data
        if np.array_equal(X, self.X_train_):
            scores = self.abof_scores_
        else:
            scores = self.decision_function(X)

        # Lower ABOF = outlier
        is_inlier = scores >= self.threshold_
        return np.where(is_inlier, 1, -1)

    def fit_predict(self, X: np.ndarray, _y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and predict anomalies in training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        _y : np.ndarray or None, default=None
            Ignored.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            -1 for anomalies, 1 for normal instances.
        """
        self.fit(X, _y)
        is_inlier = self.abof_scores_ >= self.threshold_
        return np.where(is_inlier, 1, -1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Alias for decision_function for compatibility.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            ABOF scores.
        """
        return self.decision_function(X)
