"""Shapelet transform - interpretable time-series classification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tuiml._cpp_ext import timeseries as _cpp_ts
from tuiml.algorithms.timeseries.classification._base import (
    TimeSeriesClassifier,
    as_panel,
)
from tuiml.base.algorithms import classifier


def _z_normalise(window: np.ndarray) -> np.ndarray:
    """Z-normalise a subsequence, leaving a flat one at zero.

    Parameters
    ----------
    window : np.ndarray of shape (length,)
        Subsequence to normalise.

    Returns
    -------
    normalised : np.ndarray of shape (length,)
        Zero-mean, unit-variance copy, or zeros if the input is constant.
    """
    std = window.std()
    if std <= 1e-12:
        return np.zeros_like(window)
    return (window - window.mean()) / std


def _f_statistic(distances: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Score every candidate by one-way ANOVA on its distance vector.

    A shapelet is useful when the distances it produces separate the classes,
    which is exactly a between-group versus within-group variance ratio. This
    scores every candidate at once.

    Parameters
    ----------
    distances : np.ndarray of shape (n_samples, n_candidates)
        Shapelet distance of each series to each candidate.
    y : np.ndarray of shape (n_samples,)
        Class labels.

    Returns
    -------
    scores : np.ndarray of shape (n_candidates,)
        F-statistic per candidate; larger separates the classes better.
    """
    classes = np.unique(y)
    n_samples = distances.shape[0]
    grand_mean = distances.mean(axis=0)

    between = np.zeros(distances.shape[1])
    within = np.zeros(distances.shape[1])
    for label in classes:
        group = distances[y == label]
        group_mean = group.mean(axis=0)
        between += len(group) * (group_mean - grand_mean) ** 2
        within += ((group - group_mean) ** 2).sum(axis=0)

    degrees_between = max(len(classes) - 1, 1)
    degrees_within = max(n_samples - len(classes), 1)
    # A candidate with no within-class spread is perfectly separating; the
    # floor keeps the ratio finite rather than propagating an inf.
    return (between / degrees_between) / np.maximum(
        within / degrees_within, 1e-12
    )


def _information_gain(distances: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Score every candidate by the best information gain of a distance split.

    Parameters
    ----------
    distances : np.ndarray of shape (n_samples, n_candidates)
        Shapelet distance of each series to each candidate.
    y : np.ndarray of shape (n_samples,)
        Class labels.

    Returns
    -------
    scores : np.ndarray of shape (n_candidates,)
        Best information gain achievable by thresholding each candidate's
        distances.
    """
    classes = np.unique(y)
    n_samples = len(y)
    encoded = np.searchsorted(classes, y)

    def entropy(counts: np.ndarray) -> np.ndarray:
        """Return the entropy of each column of a class-count matrix."""
        total = counts.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            proportion = np.where(total > 0, counts / np.maximum(total, 1), 0.0)
            terms = np.where(proportion > 0, -proportion * np.log2(proportion), 0.0)
        return terms.sum(axis=0)

    base_counts = np.bincount(encoded, minlength=len(classes)).astype(float)
    base_entropy = float(entropy(base_counts[:, None])[0])

    order = np.argsort(distances, axis=0)
    sorted_labels = encoded[order]

    # Cumulative class counts down the sorted order give every split at once.
    one_hot = (sorted_labels[:, :, None] == np.arange(len(classes))).astype(float)
    left_counts = np.cumsum(one_hot, axis=0)
    total_counts = left_counts[-1]

    best = np.zeros(distances.shape[1])
    for split in range(1, n_samples):
        left = left_counts[split - 1].T
        right = (total_counts - left_counts[split - 1]).T
        gain = (
            base_entropy
            - (split / n_samples) * entropy(left)
            - ((n_samples - split) / n_samples) * entropy(right)
        )
        best = np.maximum(best, gain)
    return best


@classifier(
    tags=["timeseries", "classification", "shapelet", "interpretable"],
    version="1.0.0",
)
class ShapeletTransformClassifier(TimeSeriesClassifier):
    """Classification by **which short subsequences** a series contains.

    A shapelet is a short subsequence whose presence — or absence — separates
    the classes. Where
    :class:`~tuiml.algorithms.timeseries.classification.MiniRocketClassifier`
    is accurate but opaque and
    :class:`~tuiml.algorithms.timeseries.classification.DTWNeighborsClassifier`
    compares whole series, this method answers **"what does the model actually
    look for?"** with an exhibit: the fitted shapelets are real subsequences
    from the training data that you can plot next to a series and read.

    Overview
    --------
    1. Sample candidate subsequences at random positions and lengths from
       random training series.
    2. For every candidate, compute its distance to every training series —
       the smallest z-normalised Euclidean distance to any window.
    3. Score each candidate by how well those distances separate the classes.
    4. Keep the best, discarding candidates that overlap an already-kept one.
    5. Represent each series by its distance to every kept shapelet, and fit a
       classifier on that.

    Theory
    ------
    The distance from series :math:`X` to shapelet :math:`S` of length
    :math:`m` is

    .. math::
        d(X, S) = \\min_{p} \\frac{1}{\\sqrt{m}}
        \\left\\| \\hat{z}(X_{p:p+m}) - S \\right\\|_2

    where :math:`\\hat{z}` z-normalises the window. Normalising each window
    makes the match invariant to local offset and scale — a shapelet found in
    a high-amplitude series still matches the same shape at low amplitude —
    and dividing by :math:`\\sqrt{m}` keeps shapelets of different lengths
    comparable.

    Because the shapelet is z-normalised, the squared distance collapses to
    :math:`2m - 2\\langle X_{p:p+m}, S \\rangle / \\sigma_p`, so the window
    normalisation never has to be materialised. The C++ kernel uses that,
    with running sums for :math:`\\sigma_p`.

    Exhaustive shapelet search is :math:`O(n^2 L^4)` and was the method's
    original obstacle. This class samples ``n_candidates`` instead, which is
    the standard modern remedy and costs little accuracy in practice.

    Parameters
    ----------
    n_shapelets : int, default=100
        Number of shapelets to keep. Also the number of output features.
    n_candidates : int, default=1000
        Candidates sampled before selection. More candidates means a better
        pool and proportionally more fitting time.
    min_length : int or float, default=0.05
        Shortest candidate. A float is a fraction of the series length.
    max_length : int or float, default=0.5
        Longest candidate. A float is a fraction of the series length.
    quality : {'f_stat', 'information_gain'}, default='f_stat'
        How candidates are scored. ``'f_stat'`` is a one-way ANOVA on the
        distances — vectorised over all candidates and much faster;
        ``'information_gain'`` is the classical criterion, :math:`O(n)` splits
        per candidate.
    remove_similar : bool, default=True
        Whether to discard a candidate that overlaps an already-kept shapelet
        from the same series. Without this the selection fills up with near
        duplicates of one strong pattern.
    estimator : Classifier, optional
        Head fitted on the distance features. Defaults to
        :class:`~tuiml.algorithms.linear.LogisticRegression`.
    random_state : int, optional
        Seed for candidate sampling.

    Attributes
    ----------
    shapelets_ : list of np.ndarray
        The kept shapelets, z-normalised. Plot these to see what the model
        looks for.
    shapelet_info_ : list of dict
        For each shapelet: ``series`` (training row it came from), ``start``,
        ``length``, ``channel`` and ``quality``.
    estimator_ : Classifier
        The fitted head.
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.

    Notes
    -----
    **Complexity.** Fitting is :math:`O(c \\, n \\, L \\, m)` for ``c``
    candidates, ``n`` training series of length ``L`` and shapelet length
    ``m`` — the candidate scan dominates, and it runs in the shared C++ kernel
    ``tuiml._cpp_ext.timeseries.shapelet_distances``. Transforming is
    :math:`O(k \\, n \\, L \\, m)` for ``k`` kept shapelets.

    **When to use.** Choose shapelets when someone will ask *why* — clinical,
    industrial or regulatory settings where a prediction has to be defended.
    Expect to give up some accuracy against MINIROCKET for that; if nobody
    needs the explanation, MINIROCKET is faster and usually better. Shapelets
    also suit problems where the class is defined by a **local** pattern that
    can appear anywhere in the series, which is exactly what the min-over-
    windows distance looks for.

    Multivariate panels are searched channel by channel, and each shapelet
    records the channel it came from, so the explanation stays specific.

    References
    ----------
    .. [Ye2009] Ye, L., & Keogh, E. (2009). Time Series Shapelets: A New
       Primitive for Data Mining. *ACM SIGKDD*, 947-956.
       :doi:`10.1145/1557019.1557122`
    .. [Hills2014] Hills, J., Lines, J., Baranauskas, E., Mapp, J., & Bagnall,
       A. (2014). Classification of Time Series by Shapelet Transformation.
       *Data Mining and Knowledge Discovery*, 28(4), 851-881.
       :doi:`10.1007/s10618-013-0322-1`
    .. [Bostrom2015] Bostrom, A., & Bagnall, A. (2015). Binary Shapelet
       Transform for Multiclass Time Series Classification. *DaWaK*, 257-269.
       :doi:`10.1007/978-3-319-22729-0_20`

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.classification.MiniRocketClassifier` : Faster and usually more accurate, but not interpretable.
    :class:`~tuiml.algorithms.timeseries.classification.DTWNeighborsClassifier` : Compares whole series rather than local patterns.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.classification import ShapeletTransformClassifier
    >>> rng = np.random.default_rng(0)
    >>> # Class 1 hides a triangular spike somewhere in the noise; class 0 does not.
    >>> X = rng.normal(0, 0.3, (60, 120))
    >>> y = np.array([0, 1] * 30)
    >>> spike = np.concatenate([np.linspace(0, 3, 8), np.linspace(3, 0, 8)])
    >>> for i in np.flatnonzero(y == 1):
    ...     start = rng.integers(0, 100)
    ...     X[i, start:start + 16] += spike
    >>> model = ShapeletTransformClassifier(
    ...     n_shapelets=10, n_candidates=200, random_state=0).fit(X, y)
    >>> float((model.predict(X) == y).mean())
    1.0

    The fitted shapelets are real subsequences you can inspect and plot:

    >>> len(model.shapelets_)
    10
    >>> sorted(model.shapelet_info_[0])
    ['channel', 'length', 'quality', 'series', 'start']
    """

    def __init__(
        self,
        n_shapelets: int = 100,
        n_candidates: int = 1000,
        min_length: float = 0.05,
        max_length: float = 0.5,
        quality: str = "f_stat",
        remove_similar: bool = True,
        estimator: Optional[Any] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize the shapelet transform classifier.

        Parameters
        ----------
        n_shapelets : int, default=100
            Number of shapelets to keep.
        n_candidates : int, default=1000
            Candidates sampled before selection.
        min_length : int or float, default=0.05
            Shortest candidate, absolute or as a fraction of series length.
        max_length : int or float, default=0.5
            Longest candidate, absolute or as a fraction of series length.
        quality : {'f_stat', 'information_gain'}, default='f_stat'
            Candidate scoring criterion.
        remove_similar : bool, default=True
            Discard candidates overlapping an already-kept shapelet.
        estimator : Classifier, optional
            Head fitted on the distance features.
        random_state : int, optional
            Seed for candidate sampling.
        """
        super().__init__()
        if quality not in ("f_stat", "information_gain"):
            raise ValueError(
                f"quality must be 'f_stat' or 'information_gain', got {quality!r}"
            )
        if n_shapelets < 1:
            raise ValueError(f"n_shapelets must be at least 1, got {n_shapelets}")
        if n_candidates < n_shapelets:
            raise ValueError(
                f"n_candidates ({n_candidates}) must be at least n_shapelets "
                f"({n_shapelets})"
            )
        self.n_shapelets = n_shapelets
        self.n_candidates = n_candidates
        self.min_length = min_length
        self.max_length = max_length
        self.quality = quality
        self.remove_similar = remove_similar
        self.estimator = estimator
        self.random_state = random_state

        # Fitted attributes
        self.shapelets_ = None
        self.shapelet_info_ = None
        self.estimator_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "n_shapelets": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Number of shapelets to keep, and of output features"
            },
            "n_candidates": {
                "type": "integer",
                "default": 1000,
                "minimum": 1,
                "description": "Candidate subsequences sampled before selection"
            },
            "min_length": {
                "type": "number",
                "default": 0.05,
                "description": "Shortest candidate, absolute or fraction of length"
            },
            "max_length": {
                "type": "number",
                "default": 0.5,
                "description": "Longest candidate, absolute or fraction of length"
            },
            "quality": {
                "type": "string",
                "enum": ["f_stat", "information_gain"],
                "default": "f_stat",
                "description": "Candidate scoring criterion"
            },
            "remove_similar": {
                "type": "boolean",
                "default": True,
                "description": "Discard candidates overlapping a kept shapelet"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for candidate sampling"
            },
            "estimator": {
                "type": "object",
                "default": None,
                "description": "Head fitted on the distance features"
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
            "interpretable",
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Fit: O(c*n*L*m) for c candidates, Transform: O(k*n*L*m) for k shapelets"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Ye, L. and Keogh, E., 2009. Time series shapelets: a new primitive "
            "for data mining. ACM SIGKDD.",
            "Hills, J., Lines, J., Baranauskas, E., Mapp, J. and Bagnall, A., 2014. "
            "Classification of time series by shapelet transformation. DMKD.",
            "Bostrom, A. and Bagnall, A., 2015. Binary shapelet transform for "
            "multiclass time series classification. DaWaK."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ShapeletTransformClassifier":
        """Search for shapelets and fit the head on the distance features.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Training series.
        y : np.ndarray of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : ShapeletTransformClassifier
            The fitted classifier.
        """
        panel, y = self._validate_fit(X, y)
        candidates = self._sample_candidates(panel)
        distances = self._candidate_distances(panel, candidates)

        scorer = (
            _f_statistic if self.quality == "f_stat" else _information_gain
        )
        scores = scorer(distances, y)
        keep = self._select(candidates, scores)

        self.shapelet_info_ = [
            {
                "series": candidates[i][0],
                "channel": candidates[i][1],
                "start": candidates[i][2],
                "length": candidates[i][3],
                "quality": float(scores[i]),
            }
            for i in keep
        ]
        self.shapelets_ = [candidates[i][4] for i in keep]

        features = distances[:, keep]
        self.estimator_ = self._resolve_estimator()
        self.estimator_.fit(features, y)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the shapelet-distance features of a panel.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Series to transform.

        Returns
        -------
        features : np.ndarray of shape (n_samples, n_shapelets)
            Distance from each series to each kept shapelet.
        """
        panel = self._validate_predict(X)
        return self._distances_to(panel, self.shapelets_, self.shapelet_info_)

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
        """Return class probabilities from the head.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Series to classify.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities as reported by the head.
        """
        self._check_is_fitted()
        return self.estimator_.predict_proba(self.transform(X))

    def _resolve_estimator(self) -> Any:
        """Return the head to fit on the distance features.

        Returns
        -------
        estimator : Classifier
            The caller's estimator, or a fresh LogisticRegression.
        """
        if self.estimator is not None:
            import copy

            return copy.deepcopy(self.estimator)

        from tuiml.algorithms.linear import LogisticRegression

        return LogisticRegression()

    def _length_bounds(self, n_timepoints: int) -> Tuple[int, int]:
        """Resolve the candidate length range to absolute values.

        Parameters
        ----------
        n_timepoints : int
            Series length.

        Returns
        -------
        low, high : int
            Inclusive shortest and longest candidate length.
        """
        def resolve(value: float, default: int) -> int:
            """Read a bound as a fraction when it is a float below one."""
            if isinstance(value, float) and 0.0 < value <= 1.0:
                return max(3, int(round(value * n_timepoints)))
            return int(value)

        low = resolve(self.min_length, 3)
        high = resolve(self.max_length, n_timepoints)
        low = max(3, min(low, n_timepoints))
        high = max(low, min(high, n_timepoints))
        return low, high

    def _sample_candidates(self, panel: np.ndarray) -> List[tuple]:
        """Draw random subsequences from the training panel.

        Parameters
        ----------
        panel : np.ndarray of shape (n_samples, n_channels, n_timepoints)
            Training panel.

        Returns
        -------
        candidates : list of tuple
            ``(series, channel, start, length, z_normalised_values)`` per
            candidate.
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_channels, n_timepoints = panel.shape
        low, high = self._length_bounds(n_timepoints)

        candidates = []
        for _ in range(self.n_candidates):
            series = int(rng.integers(n_samples))
            channel = int(rng.integers(n_channels))
            length = int(rng.integers(low, high + 1))
            start = int(rng.integers(0, n_timepoints - length + 1))
            values = _z_normalise(panel[series, channel, start : start + length])
            candidates.append((series, channel, start, length, values))
        return candidates

    def _candidate_distances(
        self, panel: np.ndarray, candidates: List[tuple]
    ) -> np.ndarray:
        """Compute every series' distance to every candidate.

        Parameters
        ----------
        panel : np.ndarray of shape (n_samples, n_channels, n_timepoints)
            Panel to measure.
        candidates : list of tuple
            Candidates from :meth:`_sample_candidates`.

        Returns
        -------
        distances : np.ndarray of shape (n_samples, n_candidates)
            Shapelet distances.
        """
        info = [
            {"channel": channel} for _, channel, _, _, _ in candidates
        ]
        values = [values for _, _, _, _, values in candidates]
        return self._distances_to(panel, values, info)

    def _distances_to(
        self, panel: np.ndarray, shapelets: List[np.ndarray], info: List[dict]
    ) -> np.ndarray:
        """Compute distances from a panel to a list of shapelets.

        Each shapelet is only ever compared against the channel it came from,
        so a multivariate explanation stays channel-specific.

        Parameters
        ----------
        panel : np.ndarray of shape (n_samples, n_channels, n_timepoints)
            Panel to measure.
        shapelets : list of np.ndarray
            Z-normalised shapelets.
        info : list of dict
            Per-shapelet metadata carrying at least ``channel``.

        Returns
        -------
        distances : np.ndarray of shape (n_samples, n_shapelets)
            Shapelet distances, columns in the order given.
        """
        distances = np.empty((len(panel), len(shapelets)))
        channels = np.array([item["channel"] for item in info])

        for channel in np.unique(channels):
            columns = np.flatnonzero(channels == channel)
            group = [shapelets[i] for i in columns]

            flat = np.ascontiguousarray(np.concatenate(group), dtype=np.float64)
            lengths = np.array([len(s) for s in group], dtype=np.int32)
            offsets = np.concatenate(
                [[0], np.cumsum(lengths[:-1])]
            ).astype(np.int32)

            distances[:, columns] = np.asarray(
                _cpp_ts.shapelet_distances(
                    np.ascontiguousarray(panel[:, channel, :]),
                    flat,
                    offsets,
                    lengths,
                )
            )
        return distances

    def _select(self, candidates: List[tuple], scores: np.ndarray) -> List[int]:
        """Pick the best candidates, optionally dropping overlapping ones.

        Parameters
        ----------
        candidates : list of tuple
            Candidates from :meth:`_sample_candidates`.
        scores : np.ndarray of shape (n_candidates,)
            Quality score per candidate.

        Returns
        -------
        keep : list of int
            Indices of the selected candidates, best first.
        """
        order = np.argsort(-scores)
        if not self.remove_similar:
            return list(order[: self.n_shapelets])

        keep: List[int] = []
        for index in order:
            series, channel, start, length, _ = candidates[index]
            overlaps = False
            for kept in keep:
                k_series, k_channel, k_start, k_length, _ = candidates[kept]
                if k_series != series or k_channel != channel:
                    continue
                # Two candidates from the same place in the same series say
                # the same thing; keeping both wastes a feature.
                if start < k_start + k_length and k_start < start + length:
                    overlaps = True
                    break
            if not overlaps:
                keep.append(int(index))
            if len(keep) == self.n_shapelets:
                break

        # Overlap filtering can exhaust the pool before the quota is met; top
        # up with the best remaining candidates rather than returning fewer
        # features than requested.
        if len(keep) < self.n_shapelets:
            for index in order:
                if int(index) not in keep:
                    keep.append(int(index))
                if len(keep) == self.n_shapelets:
                    break
        return keep

    def __repr__(self) -> str:
        """Return a readable representation of the classifier."""
        return (
            f"ShapeletTransformClassifier(n_shapelets={self.n_shapelets}, "
            f"n_candidates={self.n_candidates}, quality={self.quality!r})"
        )
