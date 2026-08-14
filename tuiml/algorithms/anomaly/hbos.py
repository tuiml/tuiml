"""HBOS - Histogram-Based Outlier Score."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from tuiml._cpp_ext import stats as _cpp_stats
from tuiml.base.algorithms import Classifier, anomaly_detector


@anomaly_detector(
    tags=["anomaly-detection", "histogram", "unsupervised", "fast"],
    version="1.0.0",
)
class HBOSDetector(Classifier):
    """HBOS scores outliers by **per-feature histogram density**.

    HBOS is the fastest useful detector there is: one histogram per feature,
    then a sum of log-inverse densities. Scoring is :math:`O(d)` per point with
    no distance computation and no neighbour search at all, which makes it the
    method of choice when throughput matters — streaming triage, first-pass
    filtering ahead of something more expensive, or datasets far too large for
    a quadratic detector.

    Overview
    --------
    1. Build a univariate histogram of each feature over the training data,
       with either equal-width or equal-frequency bins.
    2. Normalise each histogram to a density.
    3. Score a point by summing :math:`\\log(1 / \\text{density})` over
       features — rare bins contribute heavily, common bins barely at all.

    Theory
    ------
    With :math:`\\hat{p}_j` the fitted density of feature :math:`j`, the score
    is

    .. math::
        \\mathrm{HBOS}(x) = \\sum_{j=1}^{d}
        \\log \\left( \\frac{1}{\\hat{p}_j(x_j) + \\varepsilon} \\right)

    which is, up to sign and constants, the negative log-likelihood under a
    **naive** density model that assumes independent features. That assumption
    is exactly the trade: it buys linear-time scoring and costs the ability to
    see any anomaly defined by a *combination* of otherwise ordinary values.

    Bin choice matters more than any other decision here. Equal-width bins
    follow the classic formulation but degrade badly on skewed or heavy-tailed
    features, where nearly all mass lands in one bin; equal-frequency (dynamic)
    bins adapt to the empirical distribution and are the better default on real
    tabular data.

    Parameters
    ----------
    n_bins : int or str, default='auto'
        Number of bins per feature. ``'auto'`` uses the Birge-Rozenholc rule
        :math:`\\lceil n^{1/3} \\rceil` clipped to ``[5, 100]``, which grows
        with the sample size without overfitting small data.
    strategy : {'equal_frequency', 'equal_width'}, default='equal_frequency'
        Binning strategy. Equal-frequency bins adapt to skewed features and
        are the safer default; equal-width matches the original paper.
    contamination : float, default=0.1
        Expected proportion of outliers. Sets the decision threshold.
    tol : float, default=1e-12
        Density floor, preventing an infinite score in an empty bin.

    Attributes
    ----------
    edges_ : np.ndarray of shape (n_features, n_bins + 1)
        Fitted bin edges per feature.
    density_ : np.ndarray of shape (n_features, n_bins)
        Fitted density per bin per feature.
    n_bins_ : int
        Resolved number of bins.
    threshold_ : float
        Decision-function value separating inliers from outliers.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Notes
    -----
    **Complexity.** Training is :math:`O(n d \\log n)` for equal-frequency
    binning (a sort per feature) or :math:`O(n d)` for equal-width.
    Prediction is :math:`O(m d \\log b)` for ``b`` bins. Memory is
    :math:`O(d b)` — unlike ECOD, the training data is **not** retained, so
    the fitted model is tiny regardless of ``n``. Binning and lookup run in the
    shared C++ kernel ``tuiml._cpp_ext.stats``.

    **When to use.** Reach for HBOS when speed dominates, when the model must
    stay small, or as a cheap first stage in a cascade. Its blind spot is
    correlated features: on data where anomalies are joint rather than
    marginal, prefer
    :class:`~tuiml.algorithms.anomaly.IsolationForestDetector` or
    :class:`~tuiml.algorithms.anomaly.LocalOutlierFactorDetector`. Against
    :class:`~tuiml.algorithms.anomaly.ECODDetector`, HBOS is faster to score
    and far smaller in memory, but needs its bin count chosen and is not
    invariant to monotone transforms.

    References
    ----------
    .. [Goldstein2012] Goldstein, M., & Dengel, A. (2012). Histogram-based
       Outlier Score (HBOS): A Fast Unsupervised Anomaly Detection Algorithm.
       *KI-2012: Poster and Demo Track*, 59-63.

    See Also
    --------
    :class:`~tuiml.algorithms.anomaly.ECODDetector` : Rank-based, no bin count to choose.
    :class:`~tuiml.algorithms.anomaly.IsolationForestDetector` : Sees feature interactions.
    :class:`~tuiml.preprocessing.EqualWidthDiscretizer` : The same binning, used as a preprocessing step.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.anomaly import HBOSDetector
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(0, 1, (300, 4)), rng.normal(9, 1, (15, 4))])
    >>> detector = HBOSDetector(contamination=0.05).fit(X)
    >>> int((detector.predict(X)[-15:] == -1).sum())
    15
    >>> detector.n_bins_
    7
    """

    def __init__(
        self,
        n_bins: int | str = "auto",
        strategy: str = "equal_frequency",
        contamination: float = 0.1,
        tol: float = 1e-12,
    ):
        """Initialize the HBOS detector.

        Parameters
        ----------
        n_bins : int or str, default='auto'
            Number of bins per feature, or ``'auto'``.
        strategy : {'equal_frequency', 'equal_width'}, default='equal_frequency'
            Binning strategy.
        contamination : float, default=0.1
            Expected proportion of outliers.
        tol : float, default=1e-12
            Density floor.
        """
        super().__init__()
        if strategy not in ("equal_frequency", "equal_width"):
            raise ValueError(
                "strategy must be 'equal_frequency' or 'equal_width', got "
                f"{strategy!r}"
            )
        self.n_bins = n_bins
        self.strategy = strategy
        self.contamination = contamination
        self.tol = tol

        # Fitted attributes
        self.edges_ = None
        self.density_ = None
        self.n_bins_ = None
        self.threshold_ = None
        self.n_features_in_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "n_bins": {
                "oneOf": [
                    {"type": "integer", "minimum": 2},
                    {"type": "string", "enum": ["auto"]}
                ],
                "default": "auto",
                "description": "Number of bins per feature, or 'auto' for n^(1/3)"
            },
            "strategy": {
                "type": "string",
                "enum": ["equal_frequency", "equal_width"],
                "default": "equal_frequency",
                "description": "Binning strategy"
            },
            "contamination": {
                "type": "number",
                "default": 0.1,
                "minimum": 0.0,
                "maximum": 0.5,
                "description": "Expected proportion of outliers in the dataset"
            },
            "tol": {
                "type": "number",
                "default": 1e-12,
                "description": "Density floor preventing infinite scores"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "binary_class", "unsupervised", "anomaly_detection"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(n*d*log(n)), Prediction: O(m*d*log(b)), where b=n_bins"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Goldstein, M. and Dengel, A., 2012. Histogram-based outlier score "
            "(HBOS): A fast unsupervised anomaly detection algorithm. KI-2012."
        ]

    def fit(self, X: np.ndarray, _y: Optional[np.ndarray] = None) -> "HBOSDetector":
        """Fit one histogram per feature.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data. Labels are ignored; the method is unsupervised.
        _y : np.ndarray, optional
            Ignored, present for API consistency.

        Returns
        -------
        self : HBOSDetector
            The fitted detector.
        """
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.n_bins_ = self._resolve_n_bins(n_samples)

        if self.strategy == "equal_width":
            edges, density = _cpp_stats.equal_width_histogram(X, self.n_bins_)
        else:
            edges, density = _cpp_stats.equal_frequency_histogram(X, self.n_bins_)
        self.edges_ = np.asarray(edges)
        self.density_ = np.asarray(density)

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

    def feature_contributions(self, X: np.ndarray) -> np.ndarray:
        """Return each feature's contribution to a sample's outlier score.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        contributions : np.ndarray of shape (n_samples, n_features)
            Per-feature :math:`\\log(1 / \\text{density})`. Row sums equal the
            raw HBOS score, so the largest entries name the features that
            drove the flag.
        """
        self._check_is_fitted()
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        density = np.asarray(
            _cpp_stats.histogram_density(self.edges_, self.density_, X)
        )
        return np.log(1.0 / (density + self.tol))

    def _resolve_n_bins(self, n_samples: int) -> int:
        """Resolve the ``n_bins`` parameter to a concrete count.

        Parameters
        ----------
        n_samples : int
            Number of training samples.

        Returns
        -------
        n_bins : int
            Bin count, at least 2.
        """
        if self.n_bins == "auto":
            # Birge-Rozenholc style rule: resolution grows with n but is
            # capped so a large dataset does not end up with empty bins.
            return int(np.clip(np.ceil(n_samples ** (1.0 / 3.0)), 5, 100))
        if int(self.n_bins) < 2:
            raise ValueError(f"n_bins must be at least 2, got {self.n_bins}")
        return int(self.n_bins)

    def _outlier_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute the raw HBOS score, where higher means more anomalous.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Sum of per-feature log inverse densities.
        """
        density = np.asarray(
            _cpp_stats.histogram_density(self.edges_, self.density_, X)
        )
        return np.log(1.0 / (density + self.tol)).sum(axis=1)

    def __repr__(self) -> str:
        """Return a readable representation of the detector."""
        return (
            f"HBOSDetector(n_bins={self.n_bins!r}, strategy={self.strategy!r}, "
            f"contamination={self.contamination})"
        )
