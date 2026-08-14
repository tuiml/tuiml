"""COPOD - Copula-Based Outlier Detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from tuiml._cpp_ext import stats as _cpp_stats
from tuiml.base.algorithms import Classifier, anomaly_detector


@anomaly_detector(
    tags=["anomaly-detection", "parameter-free", "unsupervised", "probabilistic"],
    version="1.0.0",
)
class COPODDetector(Classifier):
    """COPOD scores outliers by their **empirical copula** tail probability.

    A copula separates the *dependence structure* of a distribution from its
    marginals. COPOD builds the empirical copula of the training data — each
    value replaced by its rank-based probability — and reads a point's outlier
    score off the copula's tails. Like
    :class:`~tuiml.algorithms.anomaly.ECODDetector` it is **parameter-free**
    and deterministic, and because it works on ranks it is untouched by
    monotone rescaling of any feature.

    Overview
    --------
    1. Replace every training value by its empirical CDF value, giving the
       empirical copula.
    2. For a query point, look up its left- and right-tail copula probability
       in each dimension.
    3. Take :math:`-\\log` of each and sum across dimensions to get a tail
       probability for the whole point.
    4. Report the largest of the left, right, and skewness-corrected sums.

    Theory
    ------
    Sklar's theorem says any joint distribution factors as

    .. math::
        F(x_1, \\dots, x_d) = C\\left( F_1(x_1), \\dots, F_d(x_d) \\right)

    for a copula :math:`C` carrying all the dependence. COPOD estimates the
    marginals :math:`F_j` empirically and reads the resulting **tail
    probability** as a measure of outlyingness: a point deep in the joint tail
    has a very small copula value and therefore a large :math:`-\\log`.

    The skewness correction picks, per dimension, the tail the data is
    actually skewed towards, so a right-skewed feature is not penalised for
    having a long right tail by construction.

    Parameters
    ----------
    contamination : float, default=0.1
        Expected proportion of outliers. Sets the decision threshold; it does
        not affect the scores themselves.

    Attributes
    ----------
    X_train_ : np.ndarray of shape (n_samples, n_features)
        Training data retained to evaluate the empirical copula at predict time.
    skewness_ : np.ndarray of shape (n_features,)
        Per-dimension adjusted Fisher-Pearson skewness.
    threshold_ : float
        Decision-function value separating inliers from outliers.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Notes
    -----
    **Complexity.** Training is :math:`O(n d \\log n)`, prediction
    :math:`O(m d \\log n)`, memory :math:`O(n d)`. The ranking runs in the
    shared C++ kernel ``tuiml._cpp_ext.stats.tail_probabilities``.

    **Relationship to ECOD.** These two are close relatives from the same
    group, and on most data they score almost identically — the difference is
    framing rather than mechanism, COPOD arriving at the tail sum through the
    empirical copula and ECOD through per-dimension ECDFs. The practical
    reason to prefer :class:`~tuiml.algorithms.anomaly.ECODDetector` is its
    :meth:`~tuiml.algorithms.anomaly.ECODDetector.feature_contributions`,
    which names the features responsible for a flag. COPOD is here because it
    is the more widely cited baseline and reviewers ask for it by name.

    **When to use.** Same territory as ECOD: high-dimensional tabular data,
    no tuning budget, a need for deterministic and reproducible scores. Both
    assume outlyingness shows up in the marginals; neither will find a point
    that is only strange in the joint distribution.

    References
    ----------
    .. [Li2020] Li, Z., Zhao, Y., Botta, N., Ionescu, C., & Hu, X. (2020).
       COPOD: Copula-Based Outlier Detection. *IEEE International Conference
       on Data Mining (ICDM)*, 1118-1123.
       :doi:`10.1109/ICDM50108.2020.00135`

    See Also
    --------
    :class:`~tuiml.algorithms.anomaly.ECODDetector` : The same tail idea, plus per-feature attribution.
    :class:`~tuiml.algorithms.anomaly.HBOSDetector` : Per-dimension densities instead of ranks.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.anomaly import COPODDetector
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(0, 1, (200, 3)), rng.normal(8, 1, (10, 3))])
    >>> detector = COPODDetector(contamination=0.05).fit(X)
    >>> int((detector.predict(X)[-10:] == -1).sum())
    10
    """

    def __init__(self, contamination: float = 0.1):
        """Initialize the COPOD detector.

        Parameters
        ----------
        contamination : float, default=0.1
            Expected proportion of outliers.
        """
        super().__init__()
        self.contamination = contamination

        # Fitted attributes
        self.X_train_ = None
        self.skewness_ = None
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
                "description": "Expected proportion of outliers in the dataset"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "binary_class", "unsupervised", "anomaly_detection"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(n*d*log(n)), Prediction: O(m*d*log(n)), where d=n_features"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Li, Z., Zhao, Y., Botta, N., Ionescu, C. and Hu, X., 2020. "
            "COPOD: Copula-based outlier detection. ICDM."
        ]

    def fit(self, X: np.ndarray, _y: Optional[np.ndarray] = None) -> "COPODDetector":
        """Fit the COPOD detector.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data. Labels are ignored; the method is unsupervised.
        _y : np.ndarray, optional
            Ignored, present for API consistency.

        Returns
        -------
        self : COPODDetector
            The fitted detector.
        """
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        self.X_train_ = X
        self.n_features_in_ = X.shape[1]
        self.skewness_ = np.asarray(_cpp_stats.skewness(X))

        scores = self._outlier_scores(X)
        self.threshold_ = float(
            np.percentile(-scores, 100.0 * self.contamination)
        )
        self._is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

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
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        return -self._outlier_scores(X)

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
        return np.where(self.decision_function(X) >= self.threshold_, 1, -1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Alias for decision_function for compatibility.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Anomaly scores. Lower scores indicate anomalies.
        """
        return self.decision_function(X)

    def _outlier_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute the raw COPOD score, where higher means more anomalous.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            The maximum of the left, right and skewness-corrected tail sums.
        """
        left, right = _cpp_stats.tail_probabilities(self.X_train_, X)

        left_tail = -np.log(left).sum(axis=1)
        right_tail = -np.log(right).sum(axis=1)
        corrected = -np.log(
            np.where(self.skewness_ < 0.0, left, right)
        ).sum(axis=1)

        return np.maximum.reduce([left_tail, right_tail, corrected])

    def __repr__(self) -> str:
        """Return a readable representation of the detector."""
        return f"COPODDetector(contamination={self.contamination})"
