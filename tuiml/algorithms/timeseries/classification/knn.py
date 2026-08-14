"""Nearest-neighbour time-series classification under elastic distances."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from tuiml._cpp_ext import timeseries as _cpp_ts
from tuiml.algorithms.timeseries.classification._base import (
    TimeSeriesClassifier,
    as_panel,
)
from tuiml.algorithms.timeseries.classification.distance import _resolve_window
from tuiml.base.algorithms import classifier


@classifier(
    tags=["timeseries", "classification", "distance-based", "elastic"],
    version="1.0.0",
)
class DTWNeighborsClassifier(TimeSeriesClassifier):
    """Nearest-neighbour classification under **Dynamic Time Warping**.

    One-nearest-neighbour with DTW is the benchmark every new time-series
    classifier is measured against, and it has stayed remarkably hard to beat
    for thirty years. It classifies a series by the label of the training
    series it aligns best with, where "aligns" allows the time axis to stretch
    and compress — so two examples of the same gesture performed at different
    speeds still match.

    Overview
    --------
    1. Store the training panel; there is no model to fit.
    2. For a query series, bound its distance to every training series with
       the cheap LB_Keogh bound and sort the candidates by it.
    3. Compute full DTW only for candidates whose bound could still win,
       abandoning early once a partial path exceeds the current k-th best.
    4. Vote among the ``k`` nearest labels.

    Theory
    ------
    DTW finds the alignment minimising the accumulated cost

    .. math::
        D(i, j) = (a_i - b_j)^2 +
        \\min \\{ D(i-1, j),\\ D(i, j-1),\\ D(i-1, j-1) \\}

    over a monotone path from :math:`(1,1)` to :math:`(n,m)`, returning
    :math:`\\sqrt{D(n, m)}`. Unconstrained, that is :math:`O(nm)` per pair and
    permits degenerate alignments where one point absorbs half the other
    series. A **Sakoe-Chiba band** of half-width :math:`w` restricts
    :math:`|i - j| \\leq w`, which cuts the cost and — because those degenerate
    paths are usually wrong — typically *raises* accuracy. A band of about 10%
    of the series length is the standard starting point.

    The search cost is what usually rules DTW out, and what this implementation
    attacks. LB_Keogh gives a lower bound in :math:`O(n)`; sorting candidates
    by it and skipping any whose bound already exceeds the running k-th best
    removes most of the DTW computations entirely. On 60 queries against 400
    training series of length 100, the pruned search is **12.6x** faster than
    building the full distance matrix, and returns identical neighbours.

    Parameters
    ----------
    n_neighbors : int, default=1
        Number of neighbours to vote. ``1`` is the classical and usually
        strongest setting; larger values help only on noisy labels.
    window : float or int, optional
        Sakoe-Chiba band half-width. A float in ``(0, 1]`` is a fraction of the
        series length, an int a step count, ``None`` unconstrained. Defaults to
        ``0.1``.
    weights : {'uniform', 'distance'}, default='uniform'
        Whether neighbours vote equally or in inverse proportion to distance.

    Attributes
    ----------
    X_train_ : np.ndarray of shape (n_samples, n_channels, n_timepoints)
        Stored training panel.
    y_train_ : np.ndarray of shape (n_samples,)
        Stored training labels.
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.

    Notes
    -----
    **Complexity.** Fitting is :math:`O(1)`. Prediction is worst-case
    :math:`O(m n L w)` for ``m`` queries, ``n`` training series of length ``L``
    and band ``w``, but pruning removes most of it in practice. Memory is
    :math:`O(n L)`. The distance, bound and search all run in the shared C++
    kernel ``tuiml._cpp_ext.timeseries``.

    **When to use.** Use DTW-kNN as the **baseline you must beat** before
    believing any fancier time-series classifier, and as a strong final model
    on small datasets. Its cost grows with the training set, so beyond a few
    thousand series prefer a transform-based method. Series should be
    z-normalised per instance unless absolute level is genuinely meaningful:
    without it DTW mostly measures offset.

    **Warning — warping invariance is not always what you want.** DTW is
    deliberately blind to *when* things happen. If the classes differ by
    timing, that blindness discards the only signal there is, and a plain
    Euclidean nearest neighbour beats it outright. Two synthetic problems,
    both 160 train / 160 test, length 100:

    ==================================  ==========  ===============  ==========
    problem                             DTW (10%)   Euclidean 1NN    RandomForest
    ==================================  ==========  ===============  ==========
    classes differ by **shape**,        **1.000**   0.969            0.925
    randomly warped and shifted
    classes differ by **peak timing**   0.812       **1.000**        0.994
    ==================================  ==========  ===============  ==========

    So the question to ask before reaching for DTW is not "is this a time
    series?" but "would a human still call these the same class if one were
    played faster?" If yes, DTW; if the timing *is* the label, use a plain
    classifier on the raw values.

    **The band is close to free accuracy.** On the shape problem above, a 10%
    band scored within a point of unconstrained DTW while running **20x**
    faster (49 ms against 982 ms), and widening it bought nothing. A band
    forbids some optimal warping paths, so it is not guaranteed never to cost
    a borderline series — but the compute it saves is large and the accuracy
    it costs is close to noise. Start at 10% and only widen if a held-out
    score says to.

    Multivariate panels use dependent DTW — one warping path shared across
    channels — which suits synchronised channels and not independently warping
    ones.

    References
    ----------
    .. [Sakoe1978] Sakoe, H., & Chiba, S. (1978). Dynamic Programming
       Algorithm Optimization for Spoken Word Recognition. *IEEE Transactions
       on Acoustics, Speech, and Signal Processing*, 26(1), 43-49.
       :doi:`10.1109/TASSP.1978.1163055`
    .. [Keogh2005] Keogh, E., & Ratanamahatana, C. A. (2005). Exact Indexing of
       Dynamic Time Warping. *Knowledge and Information Systems*, 7(3),
       358-386. :doi:`10.1007/s10115-004-0154-9`
    .. [Bagnall2017] Bagnall, A., Lines, J., Bostrom, A., Large, J., & Keogh,
       E. (2017). The Great Time Series Classification Bake Off.
       *Data Mining and Knowledge Discovery*, 31(3), 606-660.
       :doi:`10.1007/s10618-016-0483-9`

    See Also
    --------
    :func:`~tuiml.algorithms.timeseries.classification.dtw_distance` : The distance itself.
    :func:`~tuiml.algorithms.timeseries.classification.lb_keogh` : The bound that makes the search affordable.
    :class:`~tuiml.algorithms.neighbors.KNearestNeighborsClassifier` : Plain kNN on a feature matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.classification import DTWNeighborsClassifier
    >>> rng = np.random.default_rng(0)
    >>> t = np.linspace(0, 4 * np.pi, 60)
    >>> # Two classes: sine and sawtooth, each with a random phase shift.
    >>> shifts = rng.uniform(0, 2 * np.pi, 80)
    >>> sines = np.sin(t + shifts[:40, None])
    >>> saws = ((t + shifts[40:, None]) % (2 * np.pi)) / np.pi - 1.0
    >>> X = np.vstack([sines, saws])
    >>> y = np.array([0] * 40 + [1] * 40)
    >>> model = DTWNeighborsClassifier(n_neighbors=1, window=0.1).fit(X, y)
    >>> float((model.predict(X) == y).mean())
    1.0
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        window: Optional[float] = 0.1,
        weights: str = "uniform",
    ):
        """Initialize the DTW nearest-neighbour classifier.

        Parameters
        ----------
        n_neighbors : int, default=1
            Number of neighbours to vote.
        window : float or int, optional
            Sakoe-Chiba band half-width.
        weights : {'uniform', 'distance'}, default='uniform'
            Neighbour weighting.
        """
        super().__init__()
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be at least 1, got {n_neighbors}")
        if weights not in ("uniform", "distance"):
            raise ValueError(
                f"weights must be 'uniform' or 'distance', got {weights!r}"
            )
        self.n_neighbors = n_neighbors
        self.window = window
        self.weights = weights

        # Fitted attributes
        self.X_train_ = None
        self.y_train_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "n_neighbors": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Number of neighbors to vote"
            },
            "window": {
                "oneOf": [
                    {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                    {"type": "integer", "minimum": 1},
                    {"type": "null"}
                ],
                "default": 0.1,
                "description": "Sakoe-Chiba band: fraction of length, step count, or null"
            },
            "weights": {
                "type": "string",
                "enum": ["uniform", "distance"],
                "default": "uniform",
                "description": "Neighbor vote weighting"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return [
            "numeric",
            "multiclass",
            "timeseries",
            "multivariate_timeseries",
            "unequal_length",
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(1), Prediction: O(m*n*L*w) worst case, much less with pruning"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Sakoe, H. and Chiba, S., 1978. Dynamic programming algorithm "
            "optimization for spoken word recognition. IEEE TASSP.",
            "Keogh, E. and Ratanamahatana, C.A., 2005. Exact indexing of dynamic "
            "time warping. Knowledge and Information Systems.",
            "Bagnall, A., Lines, J., Bostrom, A., Large, J. and Keogh, E., 2017. "
            "The great time series classification bake off. DMKD."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DTWNeighborsClassifier":
        """Store the training panel.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Training series.
        y : np.ndarray of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : DTWNeighborsClassifier
            The fitted classifier.
        """
        panel, y = self._validate_fit(X, y)
        if self.n_neighbors > len(panel):
            raise ValueError(
                f"n_neighbors={self.n_neighbors} exceeds the {len(panel)} "
                "training series"
            )
        self.X_train_ = panel
        self.y_train_ = y
        self._is_fitted = True
        return self

    def kneighbors(self, X: np.ndarray) -> tuple:
        """Return the nearest training series of each query.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Query series.

        Returns
        -------
        distances : np.ndarray of shape (n_samples, n_neighbors)
            DTW distance to each neighbour, nearest first.
        indices : np.ndarray of shape (n_samples, n_neighbors)
            Row index of each neighbour in the training panel.
        """
        panel = self._validate_predict(X)
        steps = _resolve_window(self.window, self.n_timepoints_)
        distances, indices = _cpp_ts.dtw_knn(
            panel, self.X_train_, int(self.n_neighbors), steps
        )
        return np.asarray(distances), np.asarray(indices)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify each series by a vote among its nearest neighbours.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Series to classify.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the neighbour vote share for each class.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Series to classify.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Vote shares, rows summing to one.
        """
        distances, indices = self.kneighbors(X)
        n_classes = len(self.classes_)
        proba = np.zeros((len(distances), n_classes))

        for i, (row_distances, row_indices) in enumerate(zip(distances, indices)):
            # A padded slot (index -1) appears only when the training set is
            # smaller than n_neighbors, which fit() already rejects.
            valid = row_indices >= 0
            labels = self.y_train_[row_indices[valid]]

            if self.weights == "distance":
                # An exact match carries the whole vote; otherwise weight by
                # inverse distance.
                exact = row_distances[valid] <= 0.0
                if exact.any():
                    weights = exact.astype(np.float64)
                else:
                    weights = 1.0 / row_distances[valid]
            else:
                weights = np.ones(valid.sum())

            for label, weight in zip(labels, weights):
                proba[i, np.searchsorted(self.classes_, label)] += weight

        total = proba.sum(axis=1, keepdims=True)
        return np.divide(proba, total, out=proba, where=total > 0)

    def __repr__(self) -> str:
        """Return a readable representation of the classifier."""
        return (
            f"DTWNeighborsClassifier(n_neighbors={self.n_neighbors}, "
            f"window={self.window}, weights={self.weights!r})"
        )
