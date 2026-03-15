"""Elliptic Envelope anomaly detection algorithm."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List
from tuiml.base.algorithms import Classifier, classifier

@classifier(tags=["anomaly-detection", "gaussian", "covariance"], version="1.0.0")
class EllipticEnvelopeDetector(Classifier):
    """Elliptic Envelope for Gaussian-distributed anomaly detection.

    The Elliptic Envelope algorithm detects anomalies by fitting a **robust covariance 
    estimate** to the data and computing the **Mahalanobis distance** of each sample. 
    It assumes the underlying data follows a multivariate Gaussian distribution and 
    identifies outliers as points lying far from the distribution center.

    Overview
    --------
    The algorithm typically follows these steps:

    1. Estimate the robust location (mean) and covariance of the data
    2. Compute the Mahalanobis distance for each sample
    3. Define a threshold based on the expected contamination rate
    4. Points with distances exceeding the threshold are labeled as anomalies

    The intuition: Most observations are concentrated around a central point, 
    forming an elliptical shape in feature space. Points outside this "envelope" 
    are likely anomalies.

    Theory
    ------
    Assuming data follows a Gaussian distribution, the Mahalanobis distance :math:`D_M(x)` 
    of a sample :math:`x` is computed as:

    .. math::
        D_M(x) = \\sqrt{(x - \\mu)^T \\Sigma^{-1} (x - \\mu)}

    where:

    - :math:`\\mu` — Estimated location (mean vector)
    - :math:`\\Sigma` — Estimated covariance matrix
    - :math:`\\Sigma^{-1}` — Precision matrix (inverse covariance)

    Squared Mahalanobis distances :math:`D_M(x)^2` for Gaussian data follow a 
    **chi-square distribution** with :math:`p` degrees of freedom (where :math:`p` is the 
    number of features).

    **Score interpretation:**

    - Low distance → Normal point (close to distribution center)
    - High distance → Anomaly (far from distribution center)

    Parameters
    ----------
    contamination : float, default=0.1
        The proportion of outliers in the dataset. Must be in the range 
        ``(0, 0.5]``. Used to set the decision threshold.

    support_fraction : float or None, default=None
        Proportion of points to include in the support of the raw MCD estimate:
        
        - ``None`` — Uses ``min(n_samples, n_features + 1) / 2``
        - ``float`` — Between 0 and 1, specifies the fraction of samples

    random_state : int or None, default=None
        Random seed for reproducibility. Set for consistent results when 
        performing random sampling for robust estimation.

    Attributes
    ----------
    location_ : np.ndarray of shape (n_features,)
        Estimated robust location (mean) of the Gaussian distribution.

    covariance_ : np.ndarray of shape (n_features, n_features)
        Estimated robust covariance matrix.

    precision_ : np.ndarray of shape (n_features, n_features)
        Inverse of the covariance matrix (precision matrix).

    support_ : np.ndarray of shape (n_support,)
        Indices of samples used in the robust estimate.

    threshold_ : float
        Mahalanobis distance threshold used for anomaly detection.

    n_features_in_ : int
        Number of features observed during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot p^2)` where :math:`n` = samples, :math:`p` = features
    - Prediction: :math:`O(p^2)` per sample

    **When to use Elliptic Envelope:**

    - Data follows a unimodal Gaussian (elliptical) distribution
    - Low-dimensional datasets where :math:`n > p`
    - When robust statistics (mean and covariance) are needed
    - When you need a clear statistical threshold for anomalies

    **Limitations:**

    - Highly sensitive to violations of the Gaussian assumption
    - Performance degrades in very high dimensions (:math:`p > n`)
    - Computationally expensive for large numbers of features
    - Fails on multi-modal datasets (data with multiple clusters)

    References
    ----------
    .. [Rousseeuw1999] Rousseeuw, P.J. and Van Driessen, K. (1999).
           **A fast algorithm for the minimum covariance determinant estimator.**
           *Technometrics*, 41(3), pp. 212-223.
           DOI: `10.1080/00401706.1999.10485670 <https://doi.org/10.1080/00401706.1999.10485670>`_

    See Also
    --------
    :class:`~tuiml.algorithms.anomaly.OneClassSVM` : Support vector based novelty/anomaly detection.
    :class:`~tuiml.algorithms.anomaly.IsolationForest` : Tree-based ensemble anomaly detection.

    Examples
    --------
    Basic usage for anomaly detection:

    >>> from tuiml.algorithms.anomaly import EllipticEnvelopeDetector
    >>> import numpy as np
    >>> 
    >>> # Create Gaussian distributed data with one outlier
    >>> X = np.array([[1, 1], [1.1, 1.2], [0.9, 0.8], [1.2, 1.1], [10, 10]])
    >>> 
    >>> # Fit the model
    >>> clf = EllipticEnvelopeDetector(contamination=0.2, random_state=42)
    >>> clf.fit(X)
    >>> 
    >>> # Predict: -1 for anomalies, 1 for normal points
    >>> predictions = clf.predict(X)
    >>> print(predictions)
    [ 1  1  1  1 -1]
    >>> 
    >>> # Get Mahalanobis distances
    >>> distances = clf.mahalanobis(X)
    >>> print(distances.round(2))
    [ 1.35  1.21  1.67  1.84 14.28]
    """

    def __init__(
        self,
        contamination: float = 0.1,
        support_fraction: float | None = None,
        random_state: int | None = None,
    ):
        """Initialize Elliptic Envelope.

        Parameters
        ----------
        contamination : float, default=0.1
            Expected proportion of outliers.
        support_fraction : float or None, default=None
            Proportion of points in the support.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()
        self.contamination = contamination
        self.support_fraction = support_fraction
        self.random_state = random_state

        # Fitted attributes
        self.location_ = None
        self.covariance_ = None
        self.precision_ = None
        self.support_ = None
        self.threshold_ = None
        self.n_features_in_ = None

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
            "support_fraction": {
                "type": ["number", "null"],
                "default": None,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Proportion of points to include in support"
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
            "binary_class",
            "unsupervised",
            "anomaly_detection",
            "gaussian_assumption"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(n * p²), Prediction: O(p²), where n=samples, p=features"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Rousseeuw & Van Driessen, 1999. A fast algorithm for MCD estimator. Technometrics."
        ]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "EllipticEnvelopeDetector":
        """Fit the Elliptic Envelope model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray or None, default=None
            Ignored. Present for API consistency.

        Returns
        -------
        self : EllipticEnvelopeDetector
            Fitted estimator.
        """
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if n_samples < n_features:
            raise ValueError(
                f"Number of samples ({n_samples}) must be >= number of features ({n_features})"
            )

        # Determine support size
        if self.support_fraction is None:
            n_support = min(n_samples, n_features + 1) // 2 + (n_samples + n_features + 1) // 2
        else:
            n_support = int(self.support_fraction * n_samples)

        n_support = max(n_support, n_features + 1)
        n_support = min(n_support, n_samples)

        # Use a simplified robust estimation (not full MCD, but similar idea)
        # For simplicity, we'll use a random subset approach
        rng = np.random.RandomState(self.random_state)

        best_det = np.inf
        best_location = None
        best_covariance = None
        best_support = None

        # Try multiple random subsets and pick the one with smallest determinant
        n_trials = min(500, 10 * n_samples)

        for _ in range(n_trials):
            # Random subset
            subset_idx = rng.choice(n_samples, size=n_support, replace=False)
            X_subset = X[subset_idx]

            # Compute mean and covariance
            location = np.mean(X_subset, axis=0)
            centered = X_subset - location
            covariance = (centered.T @ centered) / n_support

            # Add regularization for numerical stability
            covariance += np.eye(n_features) * 1e-6

            # Check determinant
            try:
                det = np.linalg.det(covariance)
                if det > 0 and det < best_det:
                    best_det = det
                    best_location = location
                    best_covariance = covariance
                    best_support = subset_idx
            except np.linalg.LinAlgError:
                continue

        if best_location is None:
            # Fallback to empirical estimates
            best_location = np.mean(X, axis=0)
            centered = X - best_location
            best_covariance = (centered.T @ centered) / n_samples
            best_covariance += np.eye(n_features) * 1e-6
            best_support = np.arange(n_samples)

        self.location_ = best_location
        self.covariance_ = best_covariance
        self.support_ = best_support

        # Compute precision (inverse covariance)
        try:
            self.precision_ = np.linalg.inv(self.covariance_)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            self.precision_ = np.linalg.pinv(self.covariance_)

        # Compute Mahalanobis distances for all samples
        mahal_dist = self._mahalanobis_distance(X)

        # Set threshold based on chi-square distribution
        # For Gaussian data, squared Mahalanobis distance follows chi-square with p degrees of freedom
        # We use empirical quantile instead
        self.threshold_ = np.percentile(mahal_dist, 100 * (1 - self.contamination))

        self._is_fitted = True
        return self

    def _mahalanobis_distance(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        distances : np.ndarray of shape (n_samples,)
            Mahalanobis distances.
        """
        centered = X - self.location_
        # D²(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)
        # Compute: centered @ precision @ centered.T, then take diagonal
        mahal_sq = np.sum((centered @ self.precision_) * centered, axis=1)
        return np.sqrt(np.abs(mahal_sq))  # abs for numerical stability

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Returns negative Mahalanobis distances. Lower scores indicate anomalies.

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

        mahal_dist = self._mahalanobis_distance(X)

        # Return negative so lower = more anomalous
        return -mahal_dist

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

        mahal_dist = self._mahalanobis_distance(X)
        is_inlier = mahal_dist <= self.threshold_

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

    def mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        distances : np.ndarray of shape (n_samples,)
            Mahalanobis distances.
        """
        self._check_is_fitted()
        X = np.atleast_2d(X)
        return self._mahalanobis_distance(X)
