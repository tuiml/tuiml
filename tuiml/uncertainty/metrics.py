"""Metrics for calibration quality and conformal prediction validity."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "coverage_score",
    "average_set_size",
    "interval_width",
    "brier_score",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_curve",
]


def coverage_score(
    y_true: np.ndarray,
    prediction_sets: np.ndarray,
    classes: Optional[np.ndarray] = None,
) -> float:
    """Fraction of samples whose true label falls inside the prediction set.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True labels, or true values for interval predictions.
    prediction_sets : np.ndarray
        Either a boolean membership mask of shape ``(n_samples, n_classes)``
        whose columns follow ``classes``, or an interval array of shape
        ``(n_samples, 2)`` holding lower and upper bounds.
    classes : np.ndarray of shape (n_classes,), optional
        Column ordering of a boolean mask. Defaults to ``np.unique(y_true)``,
        which matches the ``classes_`` of every TuiML classifier. Pass it
        explicitly when the test split may not contain every class.

    Returns
    -------
    coverage : float
        Empirical coverage in ``[0, 1]``. Should be at least ``1 - alpha``.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import coverage_score
    >>> intervals = np.array([[0.0, 2.0], [1.0, 3.0], [5.0, 6.0]])
    >>> float(coverage_score(np.array([1.0, 4.0, 5.5]), intervals))
    0.6666666666666666
    """
    y_true = np.asarray(y_true)
    sets = np.asarray(prediction_sets)

    if sets.dtype == bool:
        # Labels need not be 0..k-1, so resolve them against the column order.
        known = np.unique(y_true) if classes is None else np.asarray(classes)
        column = np.searchsorted(known, y_true)
        if column.max(initial=0) >= sets.shape[1] or not np.all(known[column] == y_true):
            raise ValueError(
                "y_true contains labels absent from `classes`; pass the "
                "classifier's classes_ explicitly."
            )
        return float(sets[np.arange(y_true.size), column].mean())

    if sets.ndim == 2 and sets.shape[1] == 2:
        inside = (y_true >= sets[:, 0]) & (y_true <= sets[:, 1])
        return float(inside.mean())

    raise ValueError(
        "prediction_sets must be a boolean (n_samples, n_classes) mask or a "
        f"(n_samples, 2) interval array, got shape {sets.shape} of dtype {sets.dtype}"
    )


def average_set_size(prediction_sets: np.ndarray) -> float:
    """Mean number of classes per prediction set.

    Parameters
    ----------
    prediction_sets : np.ndarray of shape (n_samples, n_classes) of bool
        Boolean membership mask.

    Returns
    -------
    size : float
        Mean set size. Smaller is more informative at equal coverage.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import average_set_size
    >>> sets = np.array([[True, False, False], [True, True, False]])
    >>> float(average_set_size(sets))
    1.5
    """
    sets = np.asarray(prediction_sets, dtype=bool)
    return float(sets.sum(axis=1).mean())


def interval_width(intervals: np.ndarray) -> float:
    """Mean width of prediction intervals.

    Parameters
    ----------
    intervals : np.ndarray of shape (n_samples, 2)
        Lower and upper bounds per sample.

    Returns
    -------
    width : float
        Mean interval width. Smaller is sharper at equal coverage.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import interval_width
    >>> float(interval_width(np.array([[0.0, 2.0], [1.0, 2.0]])))
    1.5
    """
    intervals = np.asarray(intervals, dtype=np.float64)
    return float((intervals[:, 1] - intervals[:, 0]).mean())


def brier_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    pos_label: Optional[int] = None,
) -> float:
    """Mean squared error between predicted probabilities and outcomes.

    The Brier score is a strictly proper scoring rule: it is minimised only by
    the true probabilities, so it penalises both miscalibration and poor
    discrimination.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True labels.
    y_proba : np.ndarray of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities. A 1-D array is read as the probability of the
        positive class.
    pos_label : int, optional
        Label treated as positive in the binary case. Defaults to the largest
        label present.

    Returns
    -------
    score : float
        Brier score in ``[0, 2]``; lower is better. The binary case is bounded
        by ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import brier_score
    >>> float(brier_score(np.array([0, 1, 1]), np.array([0.1, 0.9, 0.8])))
    0.019999999999999993
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(y_proba, dtype=np.float64)

    if proba.ndim == 1:
        if pos_label is None:
            pos_label = np.unique(y_true)[-1]
        target = (y_true == pos_label).astype(np.float64)
        return float(np.mean((proba - target) ** 2))

    classes = np.unique(y_true)
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(y_true.size), np.searchsorted(classes, y_true)] = 1.0
    return float(np.mean(((proba - one_hot) ** 2).sum(axis=1)))


def _confidence_and_correctness(
    y_true: np.ndarray, y_proba: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce predictions to per-sample confidence and correctness.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True labels.
    y_proba : np.ndarray of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities.

    Returns
    -------
    confidence : np.ndarray of shape (n_samples,)
        Probability assigned to the predicted class.
    correct : np.ndarray of shape (n_samples,)
        1.0 where the prediction is right, 0.0 otherwise.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(y_proba, dtype=np.float64)

    if proba.ndim == 1:
        classes = np.unique(y_true)
        positive = classes[-1] if classes.size else 1
        predicted_positive = proba >= 0.5
        confidence = np.where(predicted_positive, proba, 1.0 - proba)
        correct = (predicted_positive == (y_true == positive)).astype(np.float64)
        return confidence, correct

    classes = np.unique(y_true)
    index = proba.argmax(axis=1)
    confidence = proba[np.arange(proba.shape[0]), index]
    correct = (classes[index] == y_true).astype(np.float64)
    return confidence, correct


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Weighted mean gap between confidence and accuracy across bins.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True labels.
    y_proba : np.ndarray of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities.
    n_bins : int, default=10
        Number of confidence bins.
    strategy : {'uniform', 'quantile'}, default='uniform'
        ``'uniform'`` uses equal-width bins; ``'quantile'`` uses bins holding
        an equal number of samples, which is more stable when confidences
        concentrate near one.

    Returns
    -------
    ece : float
        Expected calibration error in ``[0, 1]``; 0 means perfectly calibrated.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import expected_calibration_error
    >>> y = np.array([0, 0, 1, 1])
    >>> proba = np.array([0.0, 0.0, 1.0, 1.0])
    >>> float(expected_calibration_error(y, proba, n_bins=5))
    0.0
    """
    confidence, correct = _confidence_and_correctness(y_true, y_proba)
    edges = _bin_edges(confidence, n_bins, strategy)

    ece = 0.0
    n = confidence.size
    for lo, hi, closed_right in _iter_bins(edges):
        mask = (confidence > lo) & (confidence <= hi) if closed_right else (
            (confidence >= lo) & (confidence <= hi)
        )
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / n) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Largest gap between confidence and accuracy over all bins.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True labels.
    y_proba : np.ndarray of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities.
    n_bins : int, default=10
        Number of confidence bins.
    strategy : {'uniform', 'quantile'}, default='uniform'
        Binning strategy.

    Returns
    -------
    mce : float
        Maximum calibration error in ``[0, 1]``; reports the worst-case bin
        rather than the average, which matters for risk-sensitive deployment.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import maximum_calibration_error
    >>> y = np.array([0, 0, 1, 1])
    >>> proba = np.array([0.0, 0.0, 1.0, 1.0])
    >>> float(maximum_calibration_error(y, proba, n_bins=5))
    0.0
    """
    confidence, correct = _confidence_and_correctness(y_true, y_proba)
    edges = _bin_edges(confidence, n_bins, strategy)

    worst = 0.0
    for lo, hi, closed_right in _iter_bins(edges):
        mask = (confidence > lo) & (confidence <= hi) if closed_right else (
            (confidence >= lo) & (confidence <= hi)
        )
        if not mask.any():
            continue
        worst = max(worst, abs(correct[mask].mean() - confidence[mask].mean()))
    return float(worst)


def reliability_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin-wise mean confidence and observed accuracy for a reliability diagram.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True labels.
    y_proba : np.ndarray of shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities.
    n_bins : int, default=10
        Number of confidence bins.
    strategy : {'uniform', 'quantile'}, default='uniform'
        Binning strategy.

    Returns
    -------
    mean_confidence : np.ndarray of shape (n_nonempty_bins,)
        Mean predicted confidence per bin.
    accuracy : np.ndarray of shape (n_nonempty_bins,)
        Observed accuracy per bin.
    counts : np.ndarray of shape (n_nonempty_bins,)
        Number of samples per bin.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import reliability_curve
    >>> y = np.array([0, 1, 1, 1])
    >>> conf, acc, counts = reliability_curve(y, np.array([0.1, 0.8, 0.9, 0.95]), n_bins=2)
    >>> int(counts.sum())
    4
    """
    confidence, correct = _confidence_and_correctness(y_true, y_proba)
    edges = _bin_edges(confidence, n_bins, strategy)

    mean_confidence, accuracy, counts = [], [], []
    for lo, hi, closed_right in _iter_bins(edges):
        mask = (confidence > lo) & (confidence <= hi) if closed_right else (
            (confidence >= lo) & (confidence <= hi)
        )
        count = int(mask.sum())
        if count == 0:
            continue
        mean_confidence.append(confidence[mask].mean())
        accuracy.append(correct[mask].mean())
        counts.append(count)

    return (
        np.asarray(mean_confidence),
        np.asarray(accuracy),
        np.asarray(counts, dtype=int),
    )


def _bin_edges(confidence: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Compute confidence-bin edges.

    Parameters
    ----------
    confidence : np.ndarray of shape (n_samples,)
        Per-sample confidence values.
    n_bins : int
        Number of bins.
    strategy : {'uniform', 'quantile'}
        Binning strategy.

    Returns
    -------
    edges : np.ndarray of shape (n_bins + 1,)
        Monotone bin edges.
    """
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        edges = np.quantile(confidence, np.linspace(0.0, 1.0, n_bins + 1))
        # Duplicate quantiles collapse into empty bins; keep them unique.
        return np.unique(edges)
    raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")


def _iter_bins(edges: np.ndarray):
    """Yield ``(lower, upper, closed_right)`` triples over bin edges.

    Parameters
    ----------
    edges : np.ndarray of shape (n_bins + 1,)
        Monotone bin edges.

    Yields
    ------
    lo, hi, closed_right : float, float, bool
        Bin bounds and whether the left edge is exclusive. The first bin is
        closed on both sides so that a confidence exactly at ``edges[0]`` is
        counted.
    """
    for i in range(edges.size - 1):
        yield float(edges[i]), float(edges[i + 1]), i > 0
