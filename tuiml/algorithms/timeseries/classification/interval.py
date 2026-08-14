"""Time series forest - interval-based classification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from tuiml._cpp_ext import timeseries as _cpp_ts
from tuiml.algorithms.timeseries.classification._base import TimeSeriesClassifier
from tuiml.base.algorithms import classifier


@classifier(
    tags=["timeseries", "classification", "interval", "ensemble"],
    version="1.0.0",
)
class TimeSeriesForestClassifier(TimeSeriesClassifier):
    """Classification from **summary statistics of random intervals**.

    Some series are distinguished not by a local motif or an overall shape,
    but by what happens **in a particular stretch of time** — a device that
    draws more current during startup, a patient whose reading trends upward
    only in the third hour. The time series forest draws random intervals,
    describes each by three cheap statistics, and lets a forest decide which
    stretches matter.

    Because a tree can split on "the slope between t=40 and t=90", the fitted
    model localises *where* in time the difference lives, which none of the
    other members of this family does.

    Overview
    --------
    1. Draw ``n_intervals`` random intervals of random position and width.
    2. Describe each by its **mean**, **standard deviation** and
       **least-squares slope** against time.
    3. Concatenate into a feature vector of ``3 * n_intervals`` values.
    4. Fit a random forest on those features.

    Theory
    ------
    For an interval :math:`[a, b)` the three features are

    .. math::
        \\mu = \\frac{1}{w} \\sum_{t=a}^{b-1} x_t,
        \\quad
        \\sigma = \\sqrt{\\frac{1}{w} \\sum_{t=a}^{b-1} (x_t - \\mu)^2},
        \\quad
        \\beta = \\frac{\\mathrm{Cov}(t, x)}{\\mathrm{Var}(t)}

    with :math:`w = b - a`. Mean captures level, standard deviation captures
    activity, slope captures trend — between them a coarse but surprisingly
    effective description of a stretch of series.

    The three are computed from prefix sums of :math:`x`, :math:`x^2` and
    :math:`tx`, so an interval costs :math:`O(1)` whatever its width; because
    time is a run of consecutive integers, :math:`\\mathrm{Var}(t)` is the
    closed form :math:`(w^2 - 1)/12` and needs no accumulation at all.

    Parameters
    ----------
    n_intervals : int or str, default='sqrt'
        Number of random intervals. ``'sqrt'`` uses
        :math:`\\lceil \\sqrt{L} \\rceil`, the classical choice.
    min_interval : int, default=3
        Shortest interval. Below three points the slope is meaningless.
    n_estimators : int, default=200
        Trees in the forest.
    estimator : Classifier, optional
        Head fitted on the interval features. Defaults to
        :class:`~tuiml.algorithms.trees.RandomForestClassifier`.
    random_state : int, optional
        Seed for interval sampling and the forest.

    Attributes
    ----------
    intervals_ : np.ndarray of shape (n_intervals, 2)
        Fitted interval bounds, as ``[start, end)`` rows.
    estimator_ : Classifier
        The fitted head.
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.

    Notes
    -----
    **Complexity.** Feature extraction is :math:`O(n L)` to build the prefix
    sums plus :math:`O(n k)` for ``k`` intervals — the interval count does not
    multiply the series length. It runs in the shared C++ kernel
    ``tuiml._cpp_ext.timeseries.interval_features``. The forest then dominates.

    **When to use.** Reach for this when the discriminating information is
    *localised in time* and the series are aligned — the same phase of the
    same process across instances. It is the wrong choice when patterns drift
    in position, since an interval is fixed: there
    :class:`~tuiml.algorithms.timeseries.classification.ShapeletTransformClassifier`
    or :class:`~tuiml.algorithms.timeseries.classification.BOSSClassifier`
    search over positions instead. Like every member of the family it is
    beaten on raw accuracy by
    :class:`~tuiml.algorithms.timeseries.classification.MiniRocketClassifier`
    more often than not; its value here is a distinct, cheap, temporally
    localised view.

    Multivariate panels are handled by extracting the same intervals from every
    channel and concatenating.

    References
    ----------
    .. [Deng2013] Deng, H., Runger, G., Tuv, E., & Vladimir, M. (2013). A Time
       Series Forest for Classification and Feature Extraction. *Information
       Sciences*, 239, 142-153. :doi:`10.1016/j.ins.2013.02.030`
    .. [Middlehurst2020] Middlehurst, M., Large, J., & Bagnall, A. (2020). The
       Canonical Interval Forest (CIF) Classifier for Time Series
       Classification. *IEEE Big Data*, 188-195.
       :doi:`10.1109/BigData50022.2020.9378424`

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.classification.MiniRocketClassifier` : Usually more accurate; not temporally localised.
    :class:`~tuiml.algorithms.timeseries.classification.HIVECOTEClassifier` : Combines this view with the other four.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.classification import TimeSeriesForestClassifier
    >>> rng = np.random.default_rng(0)
    >>> # The classes differ only in the middle third of the series.
    >>> X = rng.normal(0, 1.0, (80, 90))
    >>> y = np.arange(80) % 2
    >>> X[y == 1, 30:60] += np.linspace(0, 4, 30)
    >>> model = TimeSeriesForestClassifier(n_estimators=100, random_state=0).fit(X, y)
    >>> float((model.predict(X) == y).mean())
    1.0
    """

    def __init__(
        self,
        n_intervals: Any = "sqrt",
        min_interval: int = 3,
        n_estimators: int = 200,
        estimator: Optional[Any] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize the time series forest.

        Parameters
        ----------
        n_intervals : int or str, default='sqrt'
            Number of random intervals, or ``'sqrt'``.
        min_interval : int, default=3
            Shortest interval.
        n_estimators : int, default=200
            Trees in the forest.
        estimator : Classifier, optional
            Head fitted on the interval features.
        random_state : int, optional
            Seed for interval sampling and the forest.
        """
        super().__init__()
        if min_interval < 1:
            raise ValueError(f"min_interval must be at least 1, got {min_interval}")
        if n_intervals != "sqrt" and int(n_intervals) < 1:
            raise ValueError(
                f"n_intervals must be 'sqrt' or a positive integer, got {n_intervals}"
            )
        self.n_intervals = n_intervals
        self.min_interval = min_interval
        self.n_estimators = n_estimators
        self.estimator = estimator
        self.random_state = random_state

        # Fitted attributes
        self.intervals_ = None
        self.estimator_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "n_intervals": {
                "oneOf": [
                    {"type": "integer", "minimum": 1},
                    {"type": "string", "enum": ["sqrt"]}
                ],
                "default": "sqrt",
                "description": "Number of random intervals, or 'sqrt' of series length"
            },
            "min_interval": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "description": "Shortest interval in time steps"
            },
            "n_estimators": {
                "type": "integer",
                "default": 200,
                "minimum": 1,
                "description": "Trees in the forest"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed"
            },
            "estimator": {
                "type": "object",
                "default": None,
                "description": "Head fitted on the interval features"
            },
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return [
            "numeric",
            "multiclass",
            "timeseries",
            "multivariate_timeseries",
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Features: O(n*L + n*k) for k intervals, plus the forest"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Deng, H., Runger, G., Tuv, E. and Vladimir, M., 2013. A time series "
            "forest for classification and feature extraction. Information Sciences.",
            "Middlehurst, M., Large, J. and Bagnall, A., 2020. The canonical "
            "interval forest (CIF) classifier for time series classification. "
            "IEEE Big Data."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TimeSeriesForestClassifier":
        """Draw intervals and fit the forest on their statistics.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Training series.
        y : np.ndarray of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : TimeSeriesForestClassifier
            The fitted classifier.
        """
        panel, y = self._validate_fit(X, y)
        self.intervals_ = self._sample_intervals(panel.shape[2])

        features = self._extract(panel)
        self.estimator_ = self._resolve_estimator()
        self.estimator_.fit(features, y)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the interval statistics of a panel.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Series to transform.

        Returns
        -------
        features : np.ndarray of shape (n_samples, 3 * n_intervals * n_channels)
            Mean, standard deviation and slope per interval per channel.
        """
        panel = self._validate_predict(X)
        return self._extract(panel)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify each series.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Series to classify.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted labels.
        """
        self._check_is_fitted()
        return self.estimator_.predict(self.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities from the forest.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Series to classify.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        self._check_is_fitted()
        return self.estimator_.predict_proba(self.transform(X))

    def _resolve_estimator(self) -> Any:
        """Return the head to fit on the interval features.

        Returns
        -------
        estimator : Classifier
            The caller's estimator, or a fresh RandomForestClassifier.
        """
        if self.estimator is not None:
            import copy

            return copy.deepcopy(self.estimator)

        from tuiml.algorithms.trees import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )

    def _sample_intervals(self, n_timepoints: int) -> np.ndarray:
        """Draw random interval bounds.

        Parameters
        ----------
        n_timepoints : int
            Series length.

        Returns
        -------
        intervals : np.ndarray of shape (n_intervals, 2)
            ``[start, end)`` rows.
        """
        rng = np.random.default_rng(self.random_state)

        if self.n_intervals == "sqrt":
            count = max(1, int(np.ceil(np.sqrt(n_timepoints))))
        else:
            count = int(self.n_intervals)

        minimum = min(self.min_interval, n_timepoints)
        intervals = np.empty((count, 2), dtype=np.int32)
        for i in range(count):
            start = int(rng.integers(0, n_timepoints - minimum + 1))
            width = int(rng.integers(minimum, n_timepoints - start + 1))
            intervals[i] = (start, start + width)
        return intervals

    def _extract(self, panel: np.ndarray) -> np.ndarray:
        """Compute interval statistics for every channel of a panel.

        Parameters
        ----------
        panel : np.ndarray of shape (n_samples, n_channels, n_timepoints)
            Series to describe.

        Returns
        -------
        features : np.ndarray of shape (n_samples, 3 * n_intervals * n_channels)
            Concatenated per-channel interval statistics.
        """
        starts = np.ascontiguousarray(self.intervals_[:, 0], dtype=np.int32)
        ends = np.ascontiguousarray(self.intervals_[:, 1], dtype=np.int32)

        blocks = [
            np.asarray(
                _cpp_ts.interval_features(
                    np.ascontiguousarray(panel[:, channel, :]), starts, ends
                )
            )
            for channel in range(panel.shape[1])
        ]
        return blocks[0] if len(blocks) == 1 else np.hstack(blocks)

    def __repr__(self) -> str:
        """Return a readable representation of the classifier."""
        return (
            f"TimeSeriesForestClassifier(n_intervals={self.n_intervals!r}, "
            f"n_estimators={self.n_estimators})"
        )
