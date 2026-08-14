"""Isotonic probability calibration via the pool-adjacent-violators algorithm."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from tuiml._cpp_ext import stats as _cpp_stats
from tuiml.uncertainty._base import Calibrator


class IsotonicCalibrator(Calibrator):
    """Non-parametric probability calibration by **isotonic regression**.

    Fits a **monotone, piecewise-constant** map from raw classifier scores to
    calibrated probabilities. Unlike :class:`~tuiml.uncertainty.PlattCalibrator`
    it assumes no functional form — only that a higher score should never mean a
    lower probability — so it corrects arbitrary monotone distortions.

    Overview
    --------
    1. Sort the held-out calibration scores in increasing order.
    2. Run the **pool-adjacent-violators algorithm** (PAVA) on the paired
       binary outcomes, merging any adjacent blocks that violate monotonicity
       into their weighted mean.
    3. Store the resulting step function as ``(thresholds_, values_)``.
    4. At transform time, interpolate a new score into that step function.

    Theory
    ------
    Given calibration pairs :math:`(s_i, y_i)` sorted by score, isotonic
    regression solves the constrained least-squares problem

    .. math::
        \\min_{p_1 \\leq p_2 \\leq \\dots \\leq p_n}
        \\sum_{i=1}^{n} w_i (p_i - y_i)^2

    PAVA solves this exactly in :math:`O(n)` by maintaining a stack of blocks
    with non-decreasing means; whenever a new value violates the order, the
    offending blocks are pooled into their weighted average.

    Because the fit is piecewise constant with at most :math:`n` levels,
    isotonic calibration is more expressive than a sigmoid but needs more
    calibration data — roughly 1000 samples before it beats Platt scaling.

    Parameters
    ----------
    out_of_bounds : {'clip', 'nan'}, default='clip'
        Behaviour for scores outside the calibration range. ``'clip'`` extends
        the boundary probabilities; ``'nan'`` returns ``np.nan``.
    increasing : bool, default=True
        Whether the calibration map is non-decreasing in the score. Set to
        False for scores where a *lower* value means a higher probability.

    Attributes
    ----------
    thresholds_ : np.ndarray of shape (n_blocks,)
        Score breakpoints of the fitted step function.
    values_ : np.ndarray of shape (n_blocks,)
        Calibrated probability of each block.
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.
    fitted_ : bool
        Whether :meth:`fit` has been called.

    Notes
    -----
    **Complexity.** Fitting is :math:`O(n \\log n)` (dominated by the sort;
    PAVA itself is :math:`O(n)`), transform is :math:`O(m \\log n)` via binary
    search. The PAVA step runs in the shared C++ kernel
    ``tuiml._cpp_ext.stats.pool_adjacent_violators``.

    **When to use.** Prefer isotonic when the calibration set is large
    (:math:`\\gtrsim 1000` samples) or the miscalibration is not sigmoidal —
    for example the systematic over-confidence of boosted ensembles. Prefer
    Platt scaling on small calibration sets, where isotonic overfits.

    References
    ----------
    .. [Zadrozny2002] Zadrozny, B., & Elkan, C. (2002). Transforming Classifier
       Scores into Accurate Multiclass Probability Estimates. *KDD*, 694-699.
       :doi:`10.1145/775047.775151`
    .. [Ayer1955] Ayer, M., Brunk, H. D., Ewing, G. M., Reid, W. T., &
       Silverman, E. (1955). An Empirical Distribution Function for Sampling
       with Incomplete Information. *Annals of Mathematical Statistics*,
       26(4), 641-647. :doi:`10.1214/aoms/1177728423`

    See Also
    --------
    :class:`~tuiml.uncertainty.PlattCalibrator` : Parametric sigmoid calibration.
    :class:`~tuiml.uncertainty.TemperatureScaler` : Single-parameter multiclass calibration.
    :func:`~tuiml.uncertainty.expected_calibration_error` : Measures the improvement.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import IsotonicCalibrator
    >>> scores = np.array([0.1, 0.2, 0.35, 0.4, 0.65, 0.7, 0.8, 0.95])
    >>> y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    >>> cal = IsotonicCalibrator()
    >>> proba = cal.fit_transform(scores, y)
    >>> bool(np.all(np.diff(proba) >= 0))
    True
    >>> float(cal.transform(np.array([0.9]))[0])
    1.0
    """

    def __init__(self, out_of_bounds: str = "clip", increasing: bool = True) -> None:
        """Initialise the isotonic calibrator.

        Parameters
        ----------
        out_of_bounds : {'clip', 'nan'}, default='clip'
            Behaviour for scores outside the calibration range.
        increasing : bool, default=True
            Whether the calibration map is non-decreasing.
        """
        super().__init__()
        if out_of_bounds not in ("clip", "nan"):
            raise ValueError(
                f"out_of_bounds must be 'clip' or 'nan', got {out_of_bounds!r}"
            )
        self.out_of_bounds = out_of_bounds
        self.increasing = increasing
        self.thresholds_: Optional[np.ndarray] = None
        self.values_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self,
        scores: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "IsotonicCalibrator":
        """Fit the isotonic calibration map on held-out scores.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,) or (n_samples, 2)
            Uncalibrated scores. A two-column array is read as binary
            probabilities and its positive column is used.
        y : np.ndarray of shape (n_samples,)
            True binary labels.
        sample_weight : np.ndarray of shape (n_samples,), optional
            Per-sample weights. Defaults to uniform.

        Returns
        -------
        self : IsotonicCalibrator
            The fitted calibrator.
        """
        s = self._as_positive_scores(scores)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        if self.classes_.size > 2:
            raise ValueError(
                "IsotonicCalibrator handles binary problems; got "
                f"{self.classes_.size} classes. Calibrate one-vs-rest or use "
                "TemperatureScaler."
            )
        # Map labels onto {0, 1} so PAVA fits the positive-class frequency.
        y_binary = (y == self.classes_[-1]).astype(np.float64)

        if sample_weight is None:
            w = np.ones_like(s)
        else:
            w = np.asarray(sample_weight, dtype=np.float64)

        x_sorted, fitted = _cpp_stats.isotonic_fit(s, y_binary, w, self.increasing)
        self.thresholds_, self.values_ = self._compress(x_sorted, fitted)
        self.fitted_ = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores onto calibrated probabilities.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,) or (n_samples, 2)
            Uncalibrated scores.

        Returns
        -------
        proba : np.ndarray of shape (n_samples,)
            Calibrated probability of the positive class.
        """
        self._check_is_fitted()
        s = self._as_positive_scores(scores)

        proba = np.interp(s, self.thresholds_, self.values_)
        if self.out_of_bounds == "nan":
            outside = (s < self.thresholds_[0]) | (s > self.thresholds_[-1])
            proba = np.where(outside, np.nan, proba)
        return proba

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Return two-column calibrated probabilities.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,) or (n_samples, 2)
            Uncalibrated scores.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, 2)
            Calibrated probabilities for the negative and positive class.
        """
        positive = self.transform(scores)
        return np.column_stack([1.0 - positive, positive])

    @staticmethod
    def _as_positive_scores(scores: np.ndarray) -> np.ndarray:
        """Reduce a score array to a 1-D positive-class score vector.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,) or (n_samples, 2)
            Uncalibrated scores or two-column probabilities.

        Returns
        -------
        s : np.ndarray of shape (n_samples,)
            Positive-class scores as float64.
        """
        s = np.asarray(scores, dtype=np.float64)
        if s.ndim == 2:
            if s.shape[1] != 2:
                raise ValueError(
                    "2-D scores must have exactly 2 columns for binary "
                    f"calibration, got shape {s.shape}"
                )
            s = s[:, 1]
        elif s.ndim != 1:
            raise ValueError(f"scores must be 1-D or 2-D, got {s.ndim}-D")
        return np.ascontiguousarray(s)

    @staticmethod
    def _compress(x_sorted: np.ndarray, fitted: np.ndarray) -> tuple:
        """Reduce the per-sample PAVA fit to the breakpoints of the step function.

        Parameters
        ----------
        x_sorted : np.ndarray of shape (n_samples,)
            Calibration scores in increasing order.
        fitted : np.ndarray of shape (n_samples,)
            Isotonic fit at each sorted score.

        Returns
        -------
        thresholds : np.ndarray of shape (n_blocks,)
            Score breakpoints, strictly increasing.
        values : np.ndarray of shape (n_blocks,)
            Calibrated probability of each breakpoint.
        """
        # Keep the last point of each constant block, plus the very first, so
        # np.interp reproduces the step function without redundant knots.
        keep = np.ones(x_sorted.size, dtype=bool)
        keep[1:-1] = (np.diff(fitted)[:-1] != 0) | (np.diff(fitted)[1:] != 0)

        thresholds = x_sorted[keep]
        values = fitted[keep]

        # np.interp needs strictly increasing x; collapse duplicate scores.
        unique_mask = np.ones(thresholds.size, dtype=bool)
        unique_mask[1:] = np.diff(thresholds) > 0
        return thresholds[unique_mask], values[unique_mask]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "type": "object",
            "properties": {
                "out_of_bounds": {
                    "type": "string",
                    "enum": ["clip", "nan"],
                    "default": "clip",
                    "description": "Behaviour outside the calibration range.",
                },
                "increasing": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether the calibration map is non-decreasing.",
                },
            },
        }

    def __repr__(self) -> str:
        """Return a readable representation of the calibrator."""
        return (
            f"IsotonicCalibrator(out_of_bounds={self.out_of_bounds!r}, "
            f"increasing={self.increasing})"
        )
