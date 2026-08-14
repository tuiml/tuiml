"""ECOD - Empirical Cumulative Distribution based Outlier Detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from tuiml._cpp_ext import stats as _cpp_stats
from tuiml.base.algorithms import Classifier, anomaly_detector


@anomaly_detector(
    tags=["anomaly-detection", "parameter-free", "unsupervised", "interpretable"],
    version="1.0.0",
)
class ECODDetector(Classifier):
    """ECOD detects outliers from **per-dimension empirical tail probabilities**.

    ECOD is **parameter-free**: it has nothing to tune, no distance metric, no
    neighbourhood size, no kernel. It asks one question per dimension — how far
    into the tail does this value sit? — and adds the surprise up across
    dimensions. Despite that simplicity it is at or near the top of large
    outlier-detection benchmarks, and it is the rare detector that can say
    **which feature** made a point look anomalous.

    Overview
    --------
    1. For each dimension, build the empirical CDF of the training values.
    2. For a point, read off its left-tail probability
       :math:`\\hat{F}_j(x_j)` and right-tail probability
       :math:`1 - \\hat{F}_j(x_j^-)`.
    3. Convert each to a surprise, :math:`-\\log(\\text{probability})`, and sum
       across dimensions.
    4. Score the point by the largest of three aggregates: left-tail only,
       right-tail only, and a skewness-guided choice that picks the tail each
       dimension is actually skewed towards.

    Theory
    ------
    The left and right tail probabilities of dimension :math:`j` are estimated
    as

    .. math::
        \\hat{F}_j^{-}(x) = \\frac{1}{n} \\sum_{i=1}^{n}
        \\mathbb{1}\\{X_{ij} \\leq x\\},
        \\quad
        \\hat{F}_j^{+}(x) = \\frac{1}{n} \\sum_{i=1}^{n}
        \\mathbb{1}\\{X_{ij} \\geq x\\}

    and the three aggregate scores are

    .. math::
        O^{-}(x) = -\\sum_j \\log \\hat{F}_j^{-}(x_j),
        \\quad
        O^{+}(x) = -\\sum_j \\log \\hat{F}_j^{+}(x_j),
        \\quad
        O^{a}(x) = -\\sum_j \\log \\hat{F}_j^{s_j}(x_j)

    where :math:`s_j` follows the sign of the dimension's skewness
    :math:`\\gamma_j`: a left-skewed dimension is scored on its left tail, a
    right-skewed one on its right. The final score is
    :math:`\\max(O^{-}, O^{+}, O^{a})`.

    Summing :math:`-\\log` probabilities is the independence assumption made
    explicit: it treats dimensions as independent, which is why ECOD is fast
    and dimension-scalable, and also why it cannot see an outlier that is only
    unusual in the *joint* distribution — a point at (tall, light) whose
    height and weight are each perfectly ordinary.

    Parameters
    ----------
    contamination : float, default=0.1
        Expected proportion of outliers. Sets the decision threshold; it does
        not affect the scores themselves.

    Attributes
    ----------
    X_train_ : np.ndarray of shape (n_samples, n_features)
        Training data retained to evaluate the empirical CDF at predict time.
    skewness_ : np.ndarray of shape (n_features,)
        Per-dimension adjusted Fisher-Pearson skewness, which selects the tail
        used by the skewness-guided aggregate.
    threshold_ : float
        Decision-function value separating inliers from outliers.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Notes
    -----
    **Complexity.** Training is :math:`O(n d \\log n)` — one sort per
    dimension — and prediction is :math:`O(m d \\log n)` by binary search.
    Memory is :math:`O(n d)` because the training matrix is retained. Both the
    sort and the search run in the shared C++ kernel
    ``tuiml._cpp_ext.stats.tail_probabilities``.

    **When to use.** ECOD is the right first thing to try on tabular data:
    nothing to tune, no scaling required — it is invariant to any monotone
    per-feature transform — and it scales to high dimension where
    distance-based detectors collapse. Use
    :class:`~tuiml.algorithms.anomaly.LocalOutlierFactorDetector` or
    :class:`~tuiml.algorithms.anomaly.IsolationForestDetector` instead when
    anomalies are defined by feature *interactions* rather than by extremeness
    in individual features.

    References
    ----------
    .. [Li2022] Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C., & Chen,
       G. H. (2022). ECOD: Unsupervised Outlier Detection Using Empirical
       Cumulative Distribution Functions. *IEEE Transactions on Knowledge and
       Data Engineering*, 35(12), 12181-12193.
       :doi:`10.1109/TKDE.2022.3159580`

    See Also
    --------
    :class:`~tuiml.algorithms.anomaly.COPODDetector` : The copula-framed sibling of this method.
    :class:`~tuiml.algorithms.anomaly.HBOSDetector` : Also per-dimension, but histogram-based rather than rank-based.
    :class:`~tuiml.algorithms.anomaly.IsolationForestDetector` : Sees feature interactions that ECOD cannot.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.anomaly import ECODDetector
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(0, 1, (200, 3)), rng.normal(8, 1, (10, 3))])
    >>> detector = ECODDetector(contamination=0.05).fit(X)
    >>> predictions = detector.predict(X)
    >>> int((predictions[-10:] == -1).sum())  # the injected outliers
    10

    The per-dimension contributions explain *why* a point was flagged:

    >>> contributions = detector.feature_contributions(X[-1:])
    >>> contributions.shape
    (1, 3)
    """

    def __init__(self, contamination: float = 0.1):
        """Initialize the ECOD detector.

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
            "Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C. and Chen, G.H., 2022. "
            "ECOD: Unsupervised outlier detection using empirical cumulative "
            "distribution functions. IEEE TKDE."
        ]

    def fit(self, X: np.ndarray, _y: Optional[np.ndarray] = None) -> "ECODDetector":
        """Fit the ECOD detector.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data. Labels are ignored; the method is unsupervised.
        _y : np.ndarray, optional
            Ignored, present for API consistency.

        Returns
        -------
        self : ECODDetector
            The fitted detector.
        """
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        self.X_train_ = X
        self.n_features_in_ = X.shape[1]
        self.skewness_ = np.asarray(_cpp_stats.skewness(X))

        scores = self._outlier_scores(X)
        # Higher raw score means more anomalous, but the library's
        # decision_function convention is the reverse, so the threshold is
        # placed on the negated scale.
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

    def feature_contributions(self, X: np.ndarray) -> np.ndarray:
        """Return each feature's contribution to a sample's outlier score.

        This is what makes ECOD interpretable: the total score is a plain sum
        of per-dimension surprises, so the largest entries of a row name the
        features responsible for the flag.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        contributions : np.ndarray of shape (n_samples, n_features)
            Per-dimension :math:`-\\log` tail probability, on the tail chosen by
            that dimension's skewness. Row sums equal the skewness-guided
            aggregate score.
        """
        self._check_is_fitted()
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        left, right = _cpp_stats.tail_probabilities(self.X_train_, X)
        skew_tail = np.where(self.skewness_ < 0.0, left, right)
        return -np.log(skew_tail)

    def _outlier_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute the raw ECOD score, where higher means more anomalous.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            The maximum of the left-tail, right-tail and skewness-guided
            aggregates.
        """
        left, right = _cpp_stats.tail_probabilities(self.X_train_, X)

        left_score = -np.log(left).sum(axis=1)
        right_score = -np.log(right).sum(axis=1)
        # Each dimension is scored on the tail its skew points towards.
        skew_tail = np.where(self.skewness_ < 0.0, left, right)
        skew_score = -np.log(skew_tail).sum(axis=1)

        return np.maximum.reduce([left_score, right_score, skew_score])

    def __repr__(self) -> str:
        """Return a readable representation of the detector."""
        return f"ECODDetector(contamination={self.contamination})"
