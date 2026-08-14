"""MINIROCKET - fast random convolutional features for time series."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tuiml._cpp_ext import timeseries as _cpp_ts
from tuiml.algorithms.timeseries.classification._base import (
    TimeSeriesClassifier,
    as_panel,
)
from tuiml.base.algorithms import classifier

_KERNEL_LENGTH = 9
_NUM_KERNELS = 84
_MAX_DILATIONS_PER_KERNEL = 32


def _quantiles(n: int) -> np.ndarray:
    """Return the low-discrepancy sequence MINIROCKET uses for bias quantiles.

    Successive multiples of the golden ratio, taken modulo one, spread far more
    evenly over ``[0, 1)`` than independent random draws would, so a modest
    number of biases still covers the output range without clustering.

    Parameters
    ----------
    n : int
        Number of quantiles to generate.

    Returns
    -------
    quantiles : np.ndarray of shape (n,)
        Values in ``[0, 1)``.
    """
    phi = (np.sqrt(5.0) + 1.0) / 2.0
    return np.array([((i + 1) * phi) % 1.0 for i in range(n)], dtype=np.float64)


def _fit_dilations(
    n_timepoints: int, n_features: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Choose the dilations and how many biases each one gets.

    Parameters
    ----------
    n_timepoints : int
        Series length.
    n_features : int
        Target number of output features.

    Returns
    -------
    dilations : np.ndarray of shape (n_dilations,)
        Exponentially spaced dilation factors.
    features_per_dilation : np.ndarray of shape (n_dilations,)
        Number of biases allocated to each dilation.
    """
    features_per_kernel = max(1, n_features // _NUM_KERNELS)
    true_max = min(features_per_kernel, _MAX_DILATIONS_PER_KERNEL)
    multiplier = features_per_kernel / true_max

    # Cap the exponent so the widest kernel still fits inside the series;
    # beyond that a dilation has no valid centre region left to score.
    max_exponent = np.log2(
        max(1.0, (n_timepoints - 1) / (_KERNEL_LENGTH - 1))
    )
    dilations, counts = np.unique(
        np.floor(2.0 ** np.linspace(0, max_exponent, true_max)).astype(np.int32),
        return_counts=True,
    )
    features_per_dilation = (counts * multiplier).astype(np.int32)

    # Hand any rounding shortfall to the smallest dilations, which carry the
    # highest-frequency detail.
    remainder = features_per_kernel - int(features_per_dilation.sum())
    index = 0
    while remainder > 0:
        features_per_dilation[index % len(features_per_dilation)] += 1
        remainder -= 1
        index += 1

    keep = features_per_dilation > 0
    return (
        np.ascontiguousarray(dilations[keep], dtype=np.int32),
        np.ascontiguousarray(features_per_dilation[keep], dtype=np.int32),
    )


@classifier(
    tags=["timeseries", "classification", "convolutional", "fast"],
    version="1.0.0",
)
class MiniRocketClassifier(TimeSeriesClassifier):
    """MINIROCKET: near state-of-the-art accuracy at a tiny fraction of the cost.

    MINIROCKET convolves each series with a **fixed** set of 84 dilated
    kernels and summarises every output by its **proportion of positive
    values** (PPV). The resulting few thousand features go to a plain linear
    classifier. It is almost deterministic — only the bias quantiles are
    sampled — and it reaches accuracy competitive with far more expensive
    time-series classifiers while running orders of magnitude faster.

    Overview
    --------
    1. Build the 84 fixed kernels: length 9, weights from ``{-1, 2}`` with
       exactly three positions holding 2.
    2. Pick exponentially spaced dilations that fit inside the series.
    3. For each kernel and dilation, set biases at quantiles of the actual
       convolution output on a sampled training series.
    4. Transform each series to one PPV feature per (kernel, dilation, bias).
    5. Fit a linear classifier on those features.

    Theory
    ------
    Every kernel :math:`w` has weights in :math:`\\{-1, 2\\}` with three 2s, so
    :math:`\\sum_i w_i = 3 \\cdot 2 + 6 \\cdot (-1) = 0`. A constant offset
    therefore cancels wherever the whole kernel overlaps real data — though not
    at the zero-padded edges, where it does not; see the normalisation note
    below. The feature for kernel :math:`w`, dilation :math:`d` and bias
    :math:`b` is

    .. math::
        \\mathrm{PPV}(X, w, d, b) =
        \\frac{1}{n} \\sum_{t} \\mathbb{1}\\{ (X * _d w)(t) > b \\}

    PPV, rather than the max pooling used by earlier convolutional
    transforms, is what carries most of MINIROCKET's accuracy: it measures how
    *often* a pattern matches, not merely how strongly it matches once.

    The speed comes from an algebraic trick. A kernel is :math:`-1` everywhere
    except at three positions holding :math:`+2`, so it equals the all-``-1``
    kernel plus a correction of :math:`+3` at those three positions. The
    all-``-1`` convolution is computed **once per dilation** and the nine
    possible corrections cached, after which each of the 84 kernels costs three
    vector additions rather than a fresh convolution. That is the difference
    between MINIROCKET and random-kernel ROCKET.

    Parameters
    ----------
    n_features : int, default=9996
        Approximate number of output features. Rounded down to a multiple of
        84 internally. The default follows the paper's 10,000.
    estimator : Classifier, optional
        Head fitted on the transformed features. Defaults to
        :class:`~tuiml.algorithms.linear.LogisticRegression`. Any TuiML
        classifier works; the transform is the method, the head is a choice.
    random_state : int, optional
        Seed for the training series sampled when fitting biases.

    Attributes
    ----------
    dilations_ : np.ndarray of shape (n_dilations,)
        Fitted dilation factors.
    features_per_dilation_ : np.ndarray of shape (n_dilations,)
        Biases allocated to each dilation.
    biases_ : np.ndarray of shape (n_features_,)
        Fitted bias thresholds.
    n_features_ : int
        Actual number of features produced.
    estimator_ : Classifier
        The fitted head.
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.

    Notes
    -----
    **Complexity.** Fitting the transform is :math:`O(84 \\, k \\, L)` for
    ``k`` dilations, independent of the training-set size — biases come from
    sampled series. Transforming is :math:`O(n \\, 84 \\, k \\, L)`, run in
    parallel over series by the shared C++ kernel
    ``tuiml._cpp_ext.timeseries``. The head then dominates.

    **When to use.** MINIROCKET is the right default for time-series
    classification at any scale where
    :class:`~tuiml.algorithms.timeseries.classification.DTWNeighborsClassifier`
    is too slow — which is most of them, since DTW's cost grows with the
    training set while this transform's does not. Unlike DTW it is **not**
    warping-invariant: it detects whether patterns occur, at what scale and how
    often, which is usually what distinguishes classes, but if two classes
    differ only by a global time-warp DTW remains the better tool.

    Measured on a synthetic problem where the class is the **frequency** of a
    short burst hidden at a random position, with amplitude, phase, sign and
    position randomised and each series z-normalised so no energy cue survives
    (400 train / 400 test, length 256):

    ===============  ==========  ===============
    model            accuracy    predict time
    ===============  ==========  ===============
    MINIROCKET       **0.983**   168 ms
    Euclidean 1NN    0.895       39 ms
    DTW-1NN (10%)    0.772       12 935 ms
    RandomForest     0.590       9 ms
    ===============  ==========  ===============

    DTW is both the slowest and among the least accurate here, because warping
    invariance actively smears the frequency information that defines the
    classes — the same trade-off documented on
    :class:`~tuiml.algorithms.timeseries.classification.DTWNeighborsClassifier`.
    Note also what the cost columns do as data grows: on an easier problem,
    going from 200 to 1000 training series left MINIROCKET's prediction time
    flat (83 ms to 76 ms) while DTW's rose with the training set (144 ms to
    670 ms). That asymptotic difference, not any single benchmark row, is the
    reason to reach for this first.

    Z-normalise each series first. The kernel weights sum to zero, so a
    constant offset cancels **wherever the whole kernel fits inside the
    series** — but the convolution is zero-padded at the edges, where only
    part of the kernel overlaps real data and the weights no longer cancel.
    Half the features are scored over that padded range, so offset invariance
    holds for the valid-centre features and not for the padded ones. Scale is
    never cancelled. Normalising removes both concerns.

    **Multivariate input** is handled channel-independently: the transform is
    applied to each channel and the features concatenated. That is a
    simplification of the paper's multivariate variant, which samples channel
    subsets per kernel; it costs feature-count efficiency, not correctness,
    and cross-channel interactions are left to the head.

    References
    ----------
    .. [Dempster2021] Dempster, A., Schmidt, D. F., & Webb, G. I. (2021).
       MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time
       Series Classification. *ACM SIGKDD*, 248-257.
       :doi:`10.1145/3447548.3467231`
    .. [Dempster2020] Dempster, A., Petitjean, F., & Webb, G. I. (2020).
       ROCKET: Exceptionally Fast and Accurate Time Series Classification
       Using Random Convolutional Kernels. *Data Mining and Knowledge
       Discovery*, 34(5), 1454-1495. :doi:`10.1007/s10618-020-00701-z`

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.classification.MiniRocketTransformer` : The transform without a head.
    :class:`~tuiml.algorithms.timeseries.classification.DTWNeighborsClassifier` : Warping-invariant, but cost grows with the training set.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.classification import MiniRocketClassifier
    >>> rng = np.random.default_rng(0)
    >>> t = np.linspace(0, 4 * np.pi, 64)
    >>> shifts = rng.uniform(0, 2 * np.pi, 80)
    >>> sines = np.sin(t + shifts[:40, None]) + rng.normal(0, 0.1, (40, 64))
    >>> squares = np.sign(np.sin(t + shifts[40:, None])) + rng.normal(0, 0.1, (40, 64))
    >>> X = np.vstack([sines, squares])
    >>> y = np.array([0] * 40 + [1] * 40)
    >>> model = MiniRocketClassifier(n_features=840, random_state=0).fit(X, y)
    >>> float((model.predict(X) == y).mean())
    1.0
    """

    def __init__(
        self,
        n_features: int = 9996,
        estimator: Optional[Any] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize the MINIROCKET classifier.

        Parameters
        ----------
        n_features : int, default=9996
            Approximate number of output features.
        estimator : Classifier, optional
            Head fitted on the transformed features.
        random_state : int, optional
            Seed for bias fitting.
        """
        super().__init__()
        if n_features < _NUM_KERNELS:
            raise ValueError(
                f"n_features must be at least {_NUM_KERNELS} (one per kernel), "
                f"got {n_features}"
            )
        self.n_features = n_features
        self.estimator = estimator
        self.random_state = random_state

        # Fitted attributes
        self.dilations_ = None
        self.features_per_dilation_ = None
        self.biases_ = None
        self.n_features_ = None
        self.estimator_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "n_features": {
                "type": "integer",
                "default": 9996,
                "minimum": 84,
                "description": "Approximate number of PPV features to produce"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for bias fitting"
            },
            "estimator": {
                "type": "object",
                "default": None,
                "description": "Head fitted on the transformed features"
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
        return "Transform fit: O(84*k*L), Transform: O(n*84*k*L), plus the head"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Dempster, A., Schmidt, D.F. and Webb, G.I., 2021. MINIROCKET: A very "
            "fast (almost) deterministic transform for time series classification. "
            "ACM SIGKDD.",
            "Dempster, A., Petitjean, F. and Webb, G.I., 2020. ROCKET: Exceptionally "
            "fast and accurate time series classification using random convolutional "
            "kernels. DMKD."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MiniRocketClassifier":
        """Fit the transform and the head.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Training series.
        y : np.ndarray of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : MiniRocketClassifier
            The fitted classifier.
        """
        panel, y = self._validate_fit(X, y)
        self._fit_transform_parameters(panel)

        features = self._apply(panel)
        self.n_features_ = features.shape[1]

        self.estimator_ = self._resolve_estimator()
        self.estimator_.fit(features, y)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the PPV feature matrix for a panel of series.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Series to transform.

        Returns
        -------
        features : np.ndarray of shape (n_samples, n_features_)
            One PPV feature per (kernel, dilation, bias), per channel.
        """
        panel = self._validate_predict(X)
        return self._apply(panel)

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
        """Return the head to fit on the transformed features.

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

    def _fit_transform_parameters(self, panel: np.ndarray) -> None:
        """Fit dilations and bias thresholds from the training panel.

        Parameters
        ----------
        panel : np.ndarray of shape (n_samples, n_channels, n_timepoints)
            Training panel.

        Returns
        -------
        None
        """
        n_timepoints = panel.shape[2]
        if n_timepoints < _KERNEL_LENGTH:
            raise ValueError(
                f"series of length {n_timepoints} is shorter than the kernel "
                f"length {_KERNEL_LENGTH}"
            )

        self.dilations_, self.features_per_dilation_ = _fit_dilations(
            n_timepoints, self.n_features
        )

        seed = 0 if self.random_state is None else int(self.random_state)
        quantiles = _quantiles(
            int(self.features_per_dilation_.sum()) * _NUM_KERNELS
        )
        # Biases are fitted on the first channel: the thresholds only need to
        # sit where convolution output has resolution, and channels of one
        # panel share a scale after normalisation.
        first_channel = np.ascontiguousarray(panel[:, 0, :])
        self.biases_ = np.asarray(
            _cpp_ts.minirocket_biases(
                first_channel,
                self.dilations_,
                self.features_per_dilation_,
                quantiles,
                seed,
            )
        )

    def _apply(self, panel: np.ndarray) -> np.ndarray:
        """Transform a panel, concatenating channels.

        Parameters
        ----------
        panel : np.ndarray of shape (n_samples, n_channels, n_timepoints)
            Series to transform.

        Returns
        -------
        features : np.ndarray of shape (n_samples, n_channels * n_per_channel)
            PPV features.
        """
        blocks = [
            np.asarray(
                _cpp_ts.minirocket_transform(
                    np.ascontiguousarray(panel[:, channel, :]),
                    self.dilations_,
                    self.features_per_dilation_,
                    self.biases_,
                )
            )
            for channel in range(panel.shape[1])
        ]
        return blocks[0] if len(blocks) == 1 else np.hstack(blocks)

    def __repr__(self) -> str:
        """Return a readable representation of the classifier."""
        head = (
            self.estimator_.__class__.__name__
            if self.estimator_ is not None
            else "LogisticRegression"
        )
        return f"MiniRocketClassifier(n_features={self.n_features}, estimator={head}())"


class MiniRocketTransformer(MiniRocketClassifier):
    """The MINIROCKET transform on its own, with no classifier attached.

    Use this to feed MINIROCKET features into a pipeline, a different learner,
    or a clustering or anomaly method — anywhere the transform is wanted but
    the classification head is not.

    Parameters
    ----------
    n_features : int, default=9996
        Approximate number of output features.
    random_state : int, optional
        Seed for the training series sampled when fitting biases.

    Attributes
    ----------
    dilations_ : np.ndarray of shape (n_dilations,)
        Fitted dilation factors.
    biases_ : np.ndarray of shape (n_features_,)
        Fitted bias thresholds.
    n_features_ : int
        Actual number of features produced.

    Notes
    -----
    Fitting needs no labels. :meth:`fit` accepts ``y=None``.

    See Also
    --------
    :class:`~tuiml.algorithms.timeseries.classification.MiniRocketClassifier` : The same transform with a head.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.classification import MiniRocketTransformer
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(20, 64))
    >>> transformer = MiniRocketTransformer(n_features=840, random_state=0).fit(X)
    >>> features = transformer.transform(X)
    >>> features.shape[0]
    20
    >>> bool(((features >= 0.0) & (features <= 1.0)).all())  # PPV is a proportion
    True
    """

    def __init__(
        self, n_features: int = 9996, random_state: Optional[int] = None
    ):
        """Initialize the MINIROCKET transformer.

        Parameters
        ----------
        n_features : int, default=9996
            Approximate number of output features.
        random_state : int, optional
            Seed for bias fitting.
        """
        super().__init__(
            n_features=n_features, estimator=None, random_state=random_state
        )

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "MiniRocketTransformer":
        """Fit the transform. Labels are not used.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Training series.
        y : np.ndarray, optional
            Ignored, present for API consistency.

        Returns
        -------
        self : MiniRocketTransformer
            The fitted transformer.
        """
        panel = as_panel(X)
        self.n_channels_ = panel.shape[1]
        self.n_timepoints_ = panel.shape[2]
        self._fit_transform_parameters(panel)
        self.n_features_ = int(
            self.features_per_dilation_.sum() * _NUM_KERNELS * self.n_channels_
        )
        self._is_fitted = True
        return self

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit the transform and return the features of ``X``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Training series.
        y : np.ndarray, optional
            Ignored, present for API consistency.

        Returns
        -------
        features : np.ndarray of shape (n_samples, n_features_)
            PPV features.
        """
        return self.fit(X).transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Not available: this class has no classification head.

        Parameters
        ----------
        X : np.ndarray
            Ignored.

        Raises
        ------
        NotImplementedError
            Always. Use
            :class:`~tuiml.algorithms.timeseries.classification.MiniRocketClassifier`
            to classify.
        """
        raise NotImplementedError(
            "MiniRocketTransformer only transforms. Use MiniRocketClassifier "
            "to classify, or fit your own estimator on transform(X)."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Not available: this class has no classification head.

        Parameters
        ----------
        X : np.ndarray
            Ignored.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "MiniRocketTransformer only transforms. Use MiniRocketClassifier "
            "to classify, or fit your own estimator on transform(X)."
        )

    def __repr__(self) -> str:
        """Return a readable representation of the transformer."""
        return f"MiniRocketTransformer(n_features={self.n_features})"
