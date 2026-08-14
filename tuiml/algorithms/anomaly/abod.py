"""ABOD - Angle-Based Outlier Detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from tuiml.base.algorithms import Classifier, anomaly_detector


@anomaly_detector(
    tags=["anomaly-detection", "angle-based", "unsupervised", "high-dimensional"],
    version="1.0.0",
)
class ABODDetector(Classifier):
    """ABOD detects outliers from the **variance of angles** to other points.

    In high dimension every pairwise distance converges to the same value —
    the curse of dimensionality that quietly ruins distance-based detectors.
    **Angles do not concentrate the same way.** ABOD exploits this: from a
    point *inside* a cloud, other points are scattered in every direction and
    the angles between them vary widely; from a point *outside* the cloud, all
    other points lie in roughly one direction and the angles barely vary. Low
    angle variance therefore means outlier.

    Overview
    --------
    1. For a point :math:`x`, take its ``n_neighbors`` nearest neighbours.
    2. For every pair of those neighbours, compute the angle they subtend at
       :math:`x`, weighted by the inverse of the distances involved.
    3. The score is the **variance** of those weighted cosines.
    4. Small variance means the surrounding points are all in one direction,
       which means :math:`x` sits outside the cloud.

    Theory
    ------
    The angle-based outlier factor of :math:`x` is the variance

    .. math::
        \\mathrm{ABOF}(x) = \\mathrm{Var}_{y, z}
        \\left( \\frac{\\langle y - x,\\ z - x \\rangle}
        {\\|y - x\\|^2 \\ \\|z - x\\|^2} \\right)

    over pairs :math:`y, z` drawn from the point's neighbourhood. The
    :math:`\\|\\cdot\\|^2` weighting in the denominator makes distant pairs
    count less, so the measure blends angular spread with proximity rather
    than being purely angular.

    Exact ABOD considers **all** pairs, at :math:`O(n^3)` — unusable beyond a
    few hundred points. This class implements **FastABOD**, which restricts
    the pairs to each point's ``n_neighbors`` nearest neighbours, giving
    :math:`O(n^2 d + n k^2)`. The approximation is good precisely when it
    matters: the neighbours dominate the weighted variance anyway.

    Parameters
    ----------
    n_neighbors : int, default=10
        Size of the neighbourhood whose pairs are considered. Cost grows with
        its square, so values beyond ~30 rarely pay for themselves.
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
    **Complexity.** Scoring is :math:`O(m n d)` for the neighbour search plus
    :math:`O(m k^2 d)` for the pairwise angles. Memory is :math:`O(n d)`.
    The distance matrix is computed by the shared C++ kernel
    ``tuiml._cpp_ext.distance``.

    **When to use.** ABOD earns its cost in **high dimension**, where
    :class:`~tuiml.algorithms.anomaly.KNNDetector` and
    :class:`~tuiml.algorithms.anomaly.LocalOutlierFactorDetector` degrade as
    distances concentrate. In low dimension it offers little over kNN for
    considerably more compute. As with any geometric method, scale the
    features first.

    **Warning — clustered anomalies mask each other.** ABOD assumes anomalies
    are *isolated*. When several sit together in a tight group, each one's
    nearest neighbours are the other anomalies, which surround it from all
    sides; worse, the :math:`1/\\|\\cdot\\|^2` weighting rewards a tight
    neighbourhood with a *large* factor. The group then scores as more normal
    than the genuine inliers and the ranking inverts. Measured on 300 Gaussian
    inliers in 50 dimensions with 15 anomalies placed at distance 6:

    ======================  ==========  ==========
    anomaly cluster spread  ABOD AUC    kNN AUC
    ======================  ==========  ==========
    0.05 (very tight)       0.00        1.00
    0.30                    0.00        1.00
    1.00 (as spread as      0.04        1.00
    the inliers)
    2.00                    1.00        1.00
    ======================  ==========  ==========

    The same effect shows up as the anomaly count grows: 1 or 3 isolated
    anomalies score 1.00, while 15 clustered ones score 0.19. If anomalies may
    arrive in bursts — a batch of fraudulent transactions, a stuck sensor
    emitting the same reading — use
    :class:`~tuiml.algorithms.anomaly.KNNDetector` or
    :class:`~tuiml.algorithms.anomaly.ECODDetector` instead, neither of which
    has this failure mode.

    References
    ----------
    .. [Kriegel2008] Kriegel, H.-P., Schubert, M., & Zimek, A. (2008).
       Angle-Based Outlier Detection in High-Dimensional Data. *ACM SIGKDD*,
       444-452. :doi:`10.1145/1401890.1401946`

    See Also
    --------
    :class:`~tuiml.algorithms.anomaly.KNNDetector` : Cheaper, but distance-based.
    :class:`~tuiml.algorithms.anomaly.ECODDetector` : Also dimension-scalable, at a fraction of the cost.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.anomaly import ABODDetector
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(0, 1, (200, 20)), rng.normal(6, 1, (3, 20))])
    >>> detector = ABODDetector(n_neighbors=10, contamination=0.05).fit(X)
    >>> int((detector.predict(X)[-3:] == -1).sum())  # isolated anomalies
    3

    Replacing those 3 isolated anomalies with a tight group of 15 inverts the
    ranking entirely — see the warning above before choosing this detector.
    """

    def __init__(self, n_neighbors: int = 10, contamination: float = 0.1):
        """Initialize the ABOD detector.

        Parameters
        ----------
        n_neighbors : int, default=10
            Neighbourhood size whose pairs are considered.
        contamination : float, default=0.1
            Expected proportion of outliers.
        """
        super().__init__()
        if n_neighbors < 2:
            raise ValueError(
                f"n_neighbors must be at least 2 to form a pair, got {n_neighbors}"
            )
        self.n_neighbors = n_neighbors
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
                "default": 10,
                "minimum": 2,
                "description": "Neighborhood size whose pairs form the angles"
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
        return "Training: O(1), Prediction: O(m*n*d + m*k^2*d), where k=n_neighbors"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Kriegel, H.P., Schubert, M. and Zimek, A., 2008. Angle-based outlier "
            "detection in high-dimensional data. ACM SIGKDD."
        ]

    def fit(self, X: np.ndarray, _y: Optional[np.ndarray] = None) -> "ABODDetector":
        """Fit the ABOD detector.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data. Labels are ignored; the method is unsupervised.
        _y : np.ndarray, optional
            Ignored, present for API consistency.

        Returns
        -------
        self : ABODDetector
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
        """Compute the raw ABOD score, where higher means more anomalous.

        The published factor is small for outliers, so it is negated here to
        keep the "higher is more anomalous" convention the other detectors use.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Negated angle-based outlier factor over exactly ``n_neighbors``
            distinct neighbours.
        """
        from tuiml._cpp_ext import distance as _cpp_distance

        distances = np.asarray(_cpp_distance.euclidean(X, self.X_train_))
        # One spare candidate, so a query that is itself a training point can
        # drop its own zero-distance match and still form k real neighbours.
        k = min(self.n_neighbors + 1, distances.shape[1])

        rows = np.arange(len(X))[:, None]
        neighbor_index = np.argpartition(distances, k - 1, axis=1)[:, :k]
        order = np.argsort(distances[rows, neighbor_index], axis=1)
        neighbor_index = neighbor_index[rows, order]

        variances = np.empty(len(X))
        for i in range(len(X)):
            # Vectors from the query point to each of its neighbours.
            offsets = self.X_train_[neighbor_index[i]] - X[i]
            squared_norm = np.einsum("ij,ij->i", offsets, offsets)

            # A zero-distance neighbour is the query itself, or an exact
            # duplicate of it; either way it carries no direction. Dropping
            # them first and only then truncating to k keeps the neighbourhood
            # the same size however the query relates to the training set.
            keep = squared_norm > 1e-300
            offsets = offsets[keep][: self.n_neighbors]
            squared_norm = squared_norm[keep][: self.n_neighbors]
            if offsets.shape[0] < 2:
                variances[i] = 0.0
                continue

            # Weighted cosine of every neighbour pair, as in the ABOF sum.
            gram = offsets @ offsets.T
            weights = np.outer(squared_norm, squared_norm)
            weighted = gram / weights

            # Only the off-diagonal pairs are genuine angles.
            upper = np.triu_indices(offsets.shape[0], k=1)
            variances[i] = float(np.var(weighted[upper]))

        # Small variance means outlier, so negate for the shared convention.
        return -variances

    def __repr__(self) -> str:
        """Return a readable representation of the detector."""
        return (
            f"ABODDetector(n_neighbors={self.n_neighbors}, "
            f"contamination={self.contamination})"
        )
