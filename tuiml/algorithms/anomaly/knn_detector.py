"""kNN-based outlier detection by distance to the k-th nearest neighbour."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from tuiml.base.algorithms import Classifier, anomaly_detector


@anomaly_detector(
    tags=["anomaly-detection", "distance-based", "unsupervised"],
    version="1.0.0",
)
class KNNDetector(Classifier):
    """Score outliers by their **distance to the k nearest neighbours**.

    The oldest idea in outlier detection and still one of the hardest to beat:
    a point far from its neighbours is anomalous. Unlike the per-feature
    detectors it works on the **joint** distribution, so it finds points that
    are unremarkable in every individual coordinate yet sit in an empty region
    of the space — the case where
    :class:`~tuiml.algorithms.anomaly.ECODDetector` and
    :class:`~tuiml.algorithms.anomaly.HBOSDetector` are blind.

    Overview
    --------
    1. Index the training data for nearest-neighbour search.
    2. For each point, find its ``k`` nearest training neighbours.
    3. Reduce those ``k`` distances to a single score with ``method``.
    4. Larger distance means more anomalous.

    Theory
    ------
    With :math:`d_{(1)} \\leq \\dots \\leq d_{(k)}` the sorted distances from
    :math:`x` to its ``k`` nearest neighbours, the three reductions are

    .. math::
        \\mathrm{largest}(x) = d_{(k)},
        \\quad
        \\mathrm{mean}(x) = \\frac{1}{k} \\sum_{i=1}^{k} d_{(i)},
        \\quad
        \\mathrm{median}(x) = \\mathrm{median}\\{d_{(i)}\\}

    ``'largest'`` is the classic formulation and reacts fastest to a single
    isolated point; ``'mean'`` and ``'median'`` are steadier when the data has
    small tight clusters that ``'largest'`` would flag wholesale.

    The method measures **global** distance, so it assumes one roughly uniform
    density scale. Where density varies across regions — a sparse cluster that
    is perfectly normal for its neighbourhood — a global radius mislabels the
    whole sparse region, and
    :class:`~tuiml.algorithms.anomaly.LocalOutlierFactorDetector`, which
    normalises by local density, is the correct tool.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbours ``k``. **Set this larger than the biggest group
        of anomalies you expect.** Anomalies arriving in a tight group of more
        than ``k`` points become each other's nearest neighbours, so their
        distances look small and the group masks itself; once ``k`` exceeds the
        group size the neighbourhood reaches back to the inliers and the score
        recovers. Smaller values react faster to genuinely isolated points.
    method : {'largest', 'mean', 'median'}, default='largest'
        How the ``k`` distances are reduced to one score.
    metric : {'euclidean', 'manhattan', 'cosine'}, default='euclidean'
        Distance metric.
    contamination : float, default=0.1
        Expected proportion of outliers. Sets the decision threshold.

    Attributes
    ----------
    X_train_ : np.ndarray of shape (n_samples, n_features)
        Training data retained for neighbour search.
    threshold_ : float
        Decision-function value separating inliers from outliers.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Notes
    -----
    **Complexity.** Fitting only stores the data. Scoring is
    :math:`O(m n d)` by brute force, which is the dominant cost and the
    method's real limit — the pairwise distance loop runs in the shared C++
    kernel ``tuiml._cpp_ext.distance``, but the quadratic term remains.
    Memory is :math:`O(n d)`.

    **When to use.** Use kNN when anomalies are defined by position in the
    joint space and the dataset is small enough for a quadratic scan — up to
    roughly :math:`10^4` points. Features must be scaled first: an unscaled
    feature with a large range dominates the distance and the detector
    silently becomes univariate. Above that size, or in high dimension where
    distances concentrate, prefer
    :class:`~tuiml.algorithms.anomaly.IsolationForestDetector` or the
    per-feature detectors.

    References
    ----------
    .. [Ramaswamy2000] Ramaswamy, S., Rastogi, R., & Shim, K. (2000).
       Efficient Algorithms for Mining Outliers from Large Data Sets.
       *ACM SIGMOD*, 427-438. :doi:`10.1145/342009.335437`
    .. [Angiulli2002] Angiulli, F., & Pizzuti, C. (2002). Fast Outlier
       Detection in High Dimensional Spaces. *PKDD*, 15-27.
       :doi:`10.1007/3-540-45681-3_2`

    See Also
    --------
    :class:`~tuiml.algorithms.anomaly.LocalOutlierFactorDetector` : Normalises by local density.
    :class:`~tuiml.algorithms.anomaly.ABODDetector` : Angle-based; survives higher dimension.
    :class:`~tuiml.algorithms.anomaly.ECODDetector` : Per-feature, far faster, blind to joint structure.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.anomaly import KNNDetector
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(0, 1, (200, 2)), rng.normal(7, 0.5, (10, 2))])
    >>> detector = KNNDetector(n_neighbors=15, contamination=0.05).fit(X)
    >>> int((detector.predict(X)[-10:] == -1).sum())
    10

    Note ``n_neighbors=15`` against a group of 10 anomalies. Dropping to
    ``n_neighbors=5`` lets the group mask itself and finds only 2 of them:

    >>> masked = KNNDetector(n_neighbors=5, contamination=0.05).fit(X)
    >>> int((masked.predict(X)[-10:] == -1).sum())
    2
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        method: str = "largest",
        metric: str = "euclidean",
        contamination: float = 0.1,
    ):
        """Initialize the kNN detector.

        Parameters
        ----------
        n_neighbors : int, default=5
            Number of neighbours.
        method : {'largest', 'mean', 'median'}, default='largest'
            Distance reduction.
        metric : {'euclidean', 'manhattan', 'cosine'}, default='euclidean'
            Distance metric.
        contamination : float, default=0.1
            Expected proportion of outliers.
        """
        super().__init__()
        if method not in ("largest", "mean", "median"):
            raise ValueError(
                f"method must be 'largest', 'mean' or 'median', got {method!r}"
            )
        if metric not in ("euclidean", "manhattan", "cosine"):
            raise ValueError(
                f"metric must be 'euclidean', 'manhattan' or 'cosine', got {metric!r}"
            )
        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.contamination = contamination

        # Fitted attributes
        self.X_train_ = None
        self.threshold_ = None
        self.n_features_in_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "n_neighbors": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "description": "Number of nearest neighbors to consider"
            },
            "method": {
                "type": "string",
                "enum": ["largest", "mean", "median"],
                "default": "largest",
                "description": "How the k distances are reduced to one score"
            },
            "metric": {
                "type": "string",
                "enum": ["euclidean", "manhattan", "cosine"],
                "default": "euclidean",
                "description": "Distance metric"
            },
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
        return "Training: O(1), Prediction: O(m*n*d) brute force"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Ramaswamy, S., Rastogi, R. and Shim, K., 2000. Efficient algorithms "
            "for mining outliers from large data sets. ACM SIGMOD.",
            "Angiulli, F. and Pizzuti, C., 2002. Fast outlier detection in high "
            "dimensional spaces. PKDD."
        ]

    def fit(self, X: np.ndarray, _y: Optional[np.ndarray] = None) -> "KNNDetector":
        """Fit the kNN detector.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data. Labels are ignored; the method is unsupervised.
        _y : np.ndarray, optional
            Ignored, present for API consistency.

        Returns
        -------
        self : KNNDetector
            The fitted detector.
        """
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        if self.n_neighbors >= len(X):
            raise ValueError(
                f"n_neighbors={self.n_neighbors} must be smaller than the "
                f"{len(X)} training samples"
            )
        self.X_train_ = X
        self.n_features_in_ = X.shape[1]

        # Self-matches are dropped inside _outlier_scores, so the threshold
        # is computed on the same basis as any later call.
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
        """Compute the raw kNN score, where higher means more anomalous.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Reduced neighbour distances over exactly ``n_neighbors`` distinct
            neighbours.
        """
        distances = self._pairwise(X, self.X_train_)

        # One extra candidate, so a query that is itself a training point can
        # drop its own zero-distance match and still keep k real neighbours.
        # Without this, scoring the training set would silently use k-1
        # neighbours and — for 'mean' and 'median' — average in that zero.
        k = min(self.n_neighbors + 1, distances.shape[1])
        nearest = np.sort(np.partition(distances, k - 1, axis=1)[:, :k], axis=1)

        wanted = min(self.n_neighbors, distances.shape[1])
        is_self_match = nearest[:, 0] <= 0.0
        columns = np.where(
            is_self_match[:, None],
            np.arange(1, wanted + 1),
            np.arange(0, wanted),
        )
        nearest = np.take_along_axis(nearest, np.minimum(columns, k - 1), axis=1)

        if self.method == "largest":
            return nearest[:, -1]
        if self.method == "mean":
            return nearest.mean(axis=1)
        return np.median(nearest, axis=1)

    def _pairwise(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute the pairwise distance matrix under the configured metric.

        Parameters
        ----------
        A : np.ndarray of shape (n_a, n_features)
            Query points.
        B : np.ndarray of shape (n_b, n_features)
            Reference points.

        Returns
        -------
        distances : np.ndarray of shape (n_a, n_b)
            Pairwise distances.
        """
        from tuiml._cpp_ext import distance as _cpp_distance

        kernel = {
            "euclidean": _cpp_distance.euclidean,
            "manhattan": _cpp_distance.manhattan,
            "cosine": _cpp_distance.cosine,
        }[self.metric]
        return np.asarray(kernel(A, B))

    def __repr__(self) -> str:
        """Return a readable representation of the detector."""
        return (
            f"KNNDetector(n_neighbors={self.n_neighbors}, "
            f"method={self.method!r}, metric={self.metric!r})"
        )
