"""Venn-Abers predictors: calibrated probability intervals."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from tuiml._cpp_ext import stats as _cpp_stats
from tuiml.uncertainty._base import Calibrator


class VennAbersCalibrator(Calibrator):
    """Probability **intervals** with a validity guarantee, not point estimates.

    Every other calibrator returns a single number and asks you to trust it.
    A Venn-Abers predictor returns a **pair** :math:`[p_0, p_1]` that provably
    brackets a perfectly calibrated probability — it reports its own
    calibration uncertainty. The interval is wide where calibration data is
    thin and narrow where it is plentiful.

    Overview
    --------
    1. For a test score :math:`s`, hypothetically append it to the calibration
       set **twice**: once labelled 0, once labelled 1.
    2. Run isotonic regression on each augmented set.
    3. The two fitted values at :math:`s` are :math:`p_0` and :math:`p_1`.
    4. The true calibrated probability is guaranteed to lie between them.

    Theory
    ------
    Under exchangeability, the multiprobability prediction
    :math:`\\{p_0, p_1\\}` is **valid**: one of the two is the perfectly
    calibrated probability of the test label. The width :math:`p_1 - p_0`
    shrinks as :math:`O(1/n)` in the calibration size, so it doubles as a
    diagnostic — a wide interval says the calibration set does not pin the
    probability down in that score region.

    For a single actionable number, the standard summary is

    .. math::
        p = \\frac{p_1}{1 - p_0 + p_1}

    which :meth:`transform` returns; :meth:`predict_proba_interval` returns the
    pair itself.

    Each hypothesis needs its own isotonic fit, so the implementation runs one
    PAVA pass per distinct insertion position per direction, using the shared
    C++ kernel and caching by position. Test points are deliberately *not*
    batched into a single fit: a batch of identical hypothesised labels
    perturbs the isotonic regression far more than the one point the
    definition adds, which inflates the interval.

    Parameters
    ----------
    increasing : bool, default=True
        Whether the calibration map is non-decreasing in the score.

    Attributes
    ----------
    calibration_scores_ : np.ndarray of shape (n_calibration,)
        Sorted calibration scores.
    calibration_labels_ : np.ndarray of shape (n_calibration,)
        Binary labels aligned with ``calibration_scores_``.
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.
    fitted_ : bool
        Whether :meth:`fit` has been called.

    Notes
    -----
    **Complexity.** Fitting is :math:`O(n \\log n)` — just a sort. Predicting
    is :math:`O(u n)` for :math:`u` distinct insertion positions among the
    :math:`m` test scores, since each needs its own PAVA pass; repeated
    positions are cached. The O(n \\log n) GCM formulation of Vovk et al.
    would remove the linear factor and is the natural next optimisation.

    **When to use.** Reach for Venn-Abers when a miscalibrated probability is
    expensive and you need to know *how much* to trust the calibration itself —
    medical triage, pricing, any decision with an asymmetric cost. For a plain
    point probability with less machinery, use
    :class:`~tuiml.uncertainty.IsotonicCalibrator`.

    References
    ----------
    .. [Vovk2014] Vovk, V., & Petej, I. (2014). Venn-Abers Predictors. *UAI*,
       829-838. :arxiv:`1211.0025`
    .. [Vovk2015] Vovk, V., Petej, I., & Fedorova, V. (2015). Large-Scale
       Probabilistic Predictors with and without Guarantees of Validity.
       *NeurIPS*, 892-900.

    See Also
    --------
    :class:`~tuiml.uncertainty.IsotonicCalibrator` : Point probabilities from the same PAVA kernel.
    :class:`~tuiml.uncertainty.SplitConformalClassifier` : Set-valued rather than probability-valued output.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import VennAbersCalibrator
    >>> rng = np.random.default_rng(0)
    >>> scores = rng.uniform(0, 1, 400)
    >>> y = (rng.uniform(0, 1, 400) < scores).astype(int)
    >>> va = VennAbersCalibrator().fit(scores, y)
    >>> p0, p1 = va.predict_proba_interval(np.array([0.2, 0.8]))
    >>> bool(np.all(p0 <= p1))
    True
    >>> proba = va.transform(np.array([0.2, 0.8]))
    >>> bool(proba[0] < proba[1])
    True
    """

    def __init__(self, increasing: bool = True) -> None:
        """Initialise the Venn-Abers calibrator.

        Parameters
        ----------
        increasing : bool, default=True
            Whether the calibration map is non-decreasing in the score.
        """
        super().__init__()
        self.increasing = increasing
        self.calibration_scores_: Optional[np.ndarray] = None
        self.calibration_labels_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "VennAbersCalibrator":
        """Store the calibration set that later hypotheses are appended to.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,) or (n_samples, 2)
            Uncalibrated scores.
        y : np.ndarray of shape (n_samples,)
            True binary labels.

        Returns
        -------
        self : VennAbersCalibrator
            The fitted calibrator.
        """
        s = self._as_positive_scores(scores)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        if self.classes_.size > 2:
            raise ValueError(
                "VennAbersCalibrator handles binary problems; got "
                f"{self.classes_.size} classes."
            )
        labels = (y == self.classes_[-1]).astype(np.float64)

        order = np.argsort(s, kind="stable")
        self.calibration_scores_ = np.ascontiguousarray(s[order])
        self.calibration_labels_ = np.ascontiguousarray(labels[order])
        self.fitted_ = True
        return self

    def predict_proba_interval(self, scores: np.ndarray) -> tuple:
        """Return the Venn-Abers probability interval for each score.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,) or (n_samples, 2)
            Uncalibrated scores.

        Returns
        -------
        p0 : np.ndarray of shape (n_samples,)
            Fitted probability under the hypothesis that the test label is 0.
        p1 : np.ndarray of shape (n_samples,)
            Fitted probability under the hypothesis that the test label is 1.
            Always at least ``p0``; the width reports calibration uncertainty.
        """
        self._check_is_fitted()
        s = self._as_positive_scores(scores)
        p0 = self._fit_with_hypothesis(s, 0.0)
        p1 = self._fit_with_hypothesis(s, 1.0)
        # The hypothesis-1 fit dominates the hypothesis-0 fit pointwise; clip
        # to protect against tie-breaking noise at equal scores.
        return np.minimum(p0, p1), np.maximum(p0, p1)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Collapse the probability interval to a single calibrated value.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,) or (n_samples, 2)
            Uncalibrated scores.

        Returns
        -------
        proba : np.ndarray of shape (n_samples,)
            The standard ``p1 / (1 - p0 + p1)`` summary of the interval.
        """
        p0, p1 = self.predict_proba_interval(scores)
        return p1 / (1.0 - p0 + p1)

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

    def _fit_with_hypothesis(self, scores: np.ndarray, label: float) -> np.ndarray:
        """Isotonic fit of the calibration set augmented by a hypothesised label.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,)
            Test scores to evaluate.
        label : float
            The hypothesised label, 0.0 or 1.0, appended with each test score.

        Returns
        -------
        proba : np.ndarray of shape (n_samples,)
            Fitted probability at each test score under that hypothesis.
        """
        cal_labels = self.calibration_labels_
        n_cal = cal_labels.size

        insert_at = np.searchsorted(self.calibration_scores_, scores)

        # One isotonic fit per test point, as the definition requires. Test
        # points must NOT be inserted together: a batch of identical
        # hypothesised labels perturbs the fit far more than the single point
        # the definition adds, which inflates the interval.
        # Distinct scores share a fit, so cost scales with unique scores.
        weights = np.ones(n_cal + 1)
        cache: Dict[int, float] = {}
        result = np.empty(scores.size)

        for i, position in enumerate(insert_at):
            key = int(position)
            if key not in cache:
                augmented = np.insert(cal_labels, key, label)
                fitted = _cpp_stats.pool_adjacent_violators(
                    np.ascontiguousarray(augmented), weights, self.increasing
                )
                cache[key] = float(fitted[key])
            result[i] = cache[key]

        return np.clip(result, 0.0, 1.0)

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

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "type": "object",
            "properties": {
                "increasing": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether the calibration map is non-decreasing.",
                },
            },
        }

    def __repr__(self) -> str:
        """Return a readable representation of the calibrator."""
        return f"VennAbersCalibrator(increasing={self.increasing})"
