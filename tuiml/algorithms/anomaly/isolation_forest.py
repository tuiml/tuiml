"""Isolation Forest - Unsupervised Anomaly Detection Algorithm."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List
from tuiml.base.algorithms import Classifier, classifier

@classifier(tags=["anomaly-detection", "tree-based", "unsupervised"], version="1.0.0")
class IsolationForestDetector(Classifier):
    """Isolation Forest for unsupervised anomaly detection.

    Isolation Forest is a tree-based ensemble method that detects anomalies by 
    **isolating observations**. Unlike distance-based or density-based approaches, 
    it explicitly isolates anomalies rather than profiling normal points.

    Overview
    --------
    The algorithm builds an ensemble of random isolation trees. For each tree:
    
    1. Randomly select a feature
    2. Randomly select a split value between the min and max of that feature
    3. Recursively partition the data until each point is isolated
    
    Anomalies are isolated quickly (short path lengths) because they are 
    **few** and **different** from normal instances.

    Theory
    ------
    The anomaly score for a point :math:`x` is computed as:

    .. math::
        s(x, n) = 2^{-E(h(x))/c(n)}

    where:

    - :math:`E(h(x))` — Average path length of :math:`x` over all trees
    - :math:`c(n)` — Average path length of unsuccessful search in a BST with :math:`n` samples
    - :math:`n` — Number of training samples

    **Score interpretation:**

    - Score ≈ 1.0 → Anomaly (isolated quickly)
    - Score ≈ 0.5 → Normal point
    - Score < 0.5 → Very normal (hard to isolate)

    Parameters
    ----------
    n_estimators : int, default=100
        Number of isolation trees in the ensemble. More trees generally 
        improve accuracy but increase computation time.

    max_samples : int, float, or "auto", default="auto"
        The number of samples to draw for training each tree:
        
        - ``int`` — Use exactly this many samples
        - ``float`` — Use ``max_samples * n_samples`` samples  
        - ``"auto"`` — Use ``min(256, n_samples)``

    contamination : float, default=0.1
        Expected proportion of outliers in the dataset. Used to set the 
        decision threshold. Must be in the range ``(0, 0.5]``.

    max_features : int or float, default=1.0
        Number of features to consider for each split:
        
        - ``int`` — Use exactly this many features
        - ``float`` — Use ``max_features * n_features`` features

    random_state : int or None, default=None
        Random seed for reproducibility. Set for consistent results.

    Attributes
    ----------
    trees_ : list
        Collection of fitted isolation trees.

    max_samples_ : int
        Actual number of samples used per tree after fitting.

    max_features_ : int
        Actual number of features used per tree after fitting.

    offset_ : float
        Offset for computing the decision function.

    threshold_ : float
        Decision threshold separating normal instances from anomalies.

    n_features_in_ : int
        Number of features observed during ``fit()``.

    Notes
    -----
    **Complexity:**
    
    - Training: :math:`O(t \\cdot \\psi \\cdot \\log(\\psi))` where :math:`t` = n_estimators, :math:`\\psi` = max_samples
    - Prediction: :math:`O(t \\cdot \\log(\\psi))` per sample

    **When to use Isolation Forest:**
    
    - High-dimensional datasets
    - When you don't know the contamination rate precisely
    - Streaming data (can be updated incrementally)
    - When interpretability is not the primary concern

    References
    ----------
    .. [Liu2008] Liu, F.T., Ting, K.M. and Zhou, Z.H. (2008).
           **Isolation Forest.**
           *2008 Eighth IEEE International Conference on Data Mining*, pp. 413-422.
           DOI: `10.1109/ICDM.2008.17 <https://doi.org/10.1109/ICDM.2008.17>`_
    
    .. [Liu2012] Liu, F.T., Ting, K.M. and Zhou, Z.H. (2012).
           **Isolation-based Anomaly Detection.**
           *ACM Transactions on Knowledge Discovery from Data (TKDD)*, 6(1), Article 3.
           DOI: `10.1145/2133360.2133363 <https://doi.org/10.1145/2133360.2133363>`_
    
    .. [Hariri2019] Hariri, S., Kind, M.C. and Brunner, R.J. (2019).
           **Extended Isolation Forest.**
           *IEEE Transactions on Knowledge and Data Engineering*, 33(4), pp. 1479-1489.
           DOI: `10.1109/TKDE.2019.2947676 <https://doi.org/10.1109/TKDE.2019.2947676>`_

    See Also
    --------
    :class:`~tuiml.algorithms.anomaly.ABOD` : Angle-based outlier detection for high-dimensional data.
    :class:`~tuiml.algorithms.anomaly.HalfSpaceTrees` : Streaming anomaly detection using half-space trees.

    Examples
    --------
    Basic usage for anomaly detection:

    >>> from tuiml.algorithms.anomaly import IsolationForestDetector
    >>> import numpy as np
    >>> 
    >>> # Create sample data with one anomaly
    >>> X = np.array([[1, 2], [2, 3], [3, 3], [2, 2], [10, 10]])
    >>> 
    >>> # Fit the model
    >>> clf = IsolationForestDetector(contamination=0.2, random_state=42)
    >>> clf.fit(X)
    >>> 
    >>> # Predict: -1 for anomalies, 1 for normal points
    >>> predictions = clf.predict(X)
    >>> print(predictions)
    [ 1  1  1  1 -1]
    >>> 
    >>> # Get anomaly scores (lower = more anomalous)
    >>> scores = clf.decision_function(X)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int | str = "auto",
        contamination: float = 0.1,
        max_features: int | float = 1.0,
        random_state: int | None = None,
    ):
        """Initialize Isolation Forest.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of isolation trees.
        max_samples : int, float, or "auto", default="auto"
            Number of samples per tree.
        contamination : float, default=0.1
            Expected proportion of outliers.
        max_features : int or float, default=1.0
            Number of features per tree.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state

        # Fitted attributes
        self.trees_ = None
        self.max_samples_ = None
        self.max_features_ = None
        self.offset_ = None
        self.threshold_ = None
        self.n_features_in_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "n_estimators": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Number of isolation trees in the forest"
            },
            "max_samples": {
                "oneOf": [
                    {"type": "integer", "minimum": 1},
                    {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    {"type": "string", "enum": ["auto"]}
                ],
                "default": "auto",
                "description": "Number of samples to draw for each tree"
            },
            "contamination": {
                "type": "number",
                "default": 0.1,
                "minimum": 0.0,
                "maximum": 0.5,
                "description": "Expected proportion of outliers in the dataset"
            },
            "max_features": {
                "oneOf": [
                    {"type": "integer", "minimum": 1},
                    {"type": "number", "minimum": 0.0, "maximum": 1.0}
                ],
                "default": 1.0,
                "description": "Number of features to draw for each tree"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return [
            "numeric",
            "binary_class",  # Returns -1 (anomaly) or 1 (normal)
            "unsupervised",
            "anomaly_detection"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(t * ψ * log(ψ)), Prediction: O(t * log(ψ)), where t=n_estimators, ψ=max_samples"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Liu, F.T., Ting, K.M. and Zhou, Z.H., 2008. Isolation forest. ICDM.",
            "Liu, F.T., Ting, K.M. and Zhou, Z.H., 2012. Isolation-based anomaly detection. ACM TKDD."
        ]

    def fit(self, X: np.ndarray, _y: Optional[np.ndarray] = None) -> "IsolationForestDetector":
        """Fit the Isolation Forest model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray or None, default=None
            Ignored. Present for API consistency.

        Returns
        -------
        self : IsolationForestDetector
            Fitted estimator.
        """
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Determine max_samples
        if isinstance(self.max_samples, str) and self.max_samples == "auto":
            self.max_samples_ = min(256, n_samples)
        elif isinstance(self.max_samples, int):
            self.max_samples_ = min(self.max_samples, n_samples)
        else:  # float
            self.max_samples_ = int(self.max_samples * n_samples)

        # Determine max_features
        if isinstance(self.max_features, int):
            self.max_features_ = min(self.max_features, n_features)
        else:  # float
            self.max_features_ = int(self.max_features * n_features)

        # Build trees
        rng = np.random.RandomState(self.random_state)
        self.trees_ = []

        for _ in range(self.n_estimators):
            # Sample data
            sample_indices = rng.choice(n_samples, size=self.max_samples_, replace=False)
            X_sample = X[sample_indices]

            # Build tree
            tree = self._build_tree(X_sample, 0, self.max_samples_, rng)
            self.trees_.append(tree)

        # Mark as fitted before computing threshold
        self._is_fitted = True

        # Compute offset and threshold
        scores = self.decision_function(X)
        self.offset_ = -0.5
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))

        return self

    def _build_tree(
        self,
        X: np.ndarray,
        current_height: int,
        height_limit: int,
        rng: np.random.RandomState
    ) -> Dict[str, Any]:
        """Recursively build an isolation tree.

        Parameters
        ----------
        X : np.ndarray
            Subsample to split.
        current_height : int
            Current depth in the tree.
        height_limit : int
            Maximum tree depth.
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        tree : dict
            Tree node with split information or size for leaf.
        """
        n_samples, n_features = X.shape

        # Terminal conditions
        if current_height >= height_limit or n_samples <= 1:
            return {"type": "leaf", "size": n_samples}

        # Check if all samples are identical
        if np.all(X == X[0]):
            return {"type": "leaf", "size": n_samples}

        # Select random feature
        feature_indices = rng.choice(n_features, size=self.max_features_, replace=False)

        # Find a feature with variation
        split_feature = None
        for feat_idx in feature_indices:
            col = X[:, feat_idx]
            if np.ptp(col) > 0:  # Range > 0
                split_feature = feat_idx
                break

        if split_feature is None:
            return {"type": "leaf", "size": n_samples}

        # Random split point
        col = X[:, split_feature]
        min_val, max_val = col.min(), col.max()
        split_value = rng.uniform(min_val, max_val)

        # Split data
        left_mask = col < split_value
        right_mask = ~left_mask

        if not np.any(left_mask) or not np.any(right_mask):
            return {"type": "leaf", "size": n_samples}

        # Build subtrees
        left_tree = self._build_tree(X[left_mask], current_height + 1, height_limit, rng)
        right_tree = self._build_tree(X[right_mask], current_height + 1, height_limit, rng)

        return {
            "type": "internal",
            "feature": split_feature,
            "value": split_value,
            "left": left_tree,
            "right": right_tree
        }

    def _path_length(self, x: np.ndarray, tree: Dict[str, Any], current_height: int) -> float:
        """Compute path length for a single sample in a tree.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            Single sample.
        tree : dict
            Tree structure.
        current_height : int
            Current depth.

        Returns
        -------
        path_length : float
            Path length to leaf.
        """
        if tree["type"] == "leaf":
            # Add average path length for unsuccessful search
            size = tree["size"]
            if size <= 1:
                return current_height
            # c(n) = 2H(n-1) - 2(n-1)/n, where H is harmonic number
            # Approximation: c(n) ≈ 2 * (log(n-1) + 0.5772) - 2*(n-1)/n
            c_n = 2 * (np.log(size - 1) + 0.5772156649) - 2 * (size - 1) / size if size > 2 else 1.0
            return current_height + c_n

        # Internal node
        if x[tree["feature"]] < tree["value"]:
            return self._path_length(x, tree["left"], current_height + 1)
        else:
            return self._path_length(x, tree["right"], current_height + 1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        The anomaly score is the opposite of the original paper's definition.
        Lower scores indicate anomalies.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Anomaly scores. Lower scores indicate anomalies.
        """
        self._check_is_fitted()
        X = np.atleast_2d(X)

        # Compute average path length for each sample
        avg_path_lengths = np.zeros(len(X))

        for i, x in enumerate(X):
            path_lengths = [self._path_length(x, tree, 0) for tree in self.trees_]
            avg_path_lengths[i] = np.mean(path_lengths)

        # Compute c(n) for normalization
        n = self.max_samples_
        c_n = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n if n > 2 else 1.0

        # Compute anomaly scores: s(x, n) = 2^(-E(h(x))/c(n))
        # We return the negative to make lower scores = anomalies
        scores = 2 ** (-avg_path_lengths / c_n)

        # Return negative to match convention (lower = more anomalous)
        return -scores + 0.5  # Shift so normal points are near 0

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
        scores = self.decision_function(X)

        # Use fitted threshold
        if self.threshold_ is not None:
            is_inlier = scores >= self.threshold_
        else:
            # Fallback: use 0 as threshold
            is_inlier = scores >= 0

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
            Anomaly scores.
        """
        return self.decision_function(X)
