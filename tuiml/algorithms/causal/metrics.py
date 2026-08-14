"""Evaluation metrics for uplift models.

Unlike a regression score, an uplift model is judged by how well it *ranks*
individuals by their treatment effect: the treated individuals it places at the
top should be the ones who gain the most from treatment. The Qini curve and its
area (AUUC) measure exactly that.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from tuiml.algorithms.causal.meta_learners import _check_arrays


def _coerce_rank_inputs(uplift, treatment, y):
    """Validate ``(uplift, treatment, y)`` and return sorted numeric arrays.

    Parameters
    ----------
    uplift : array-like of shape (n_samples,)
        Predicted treatment effect for each sample.
    treatment : array-like of shape (n_samples,)
        Binary treatment indicator.
    y : array-like of shape (n_samples,)
        Numeric outcome.

    Returns
    -------
    treatment : np.ndarray of shape (n_samples,) of int
        Treatment indicator.
    y : np.ndarray of shape (n_samples,)
        Outcome.
    order : np.ndarray of shape (n_samples,) of int
        Indices sorting ``uplift`` in descending order.
    """
    uplift = np.asarray(uplift, dtype=float)
    if uplift.ndim != 1:
        uplift = np.ravel(uplift)

    # _check_arrays validates treatment/y and returns coerced copies; it does
    # not touch uplift, so lengths are re-checked against it here.
    _, treatment, y = _check_arrays(np.zeros((uplift.shape[0], 1)), treatment, y)

    order = np.argsort(-uplift, kind="mergesort")
    return treatment[order], y[order]


def qini_curve(uplift, treatment, y, normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Qini curve: cumulative incremental gain when treating the
    top-ranked individuals.

    Samples are ranked by predicted uplift (highest first). At each prefix of
    the ranking, the incremental gain is the outcome of the treated samples
    minus the outcome those samples would have produced as controls, where the
    control counterfactual is estimated by scaling the control outcomes by the
    global treatment ratio :math:`n_t / n_c`.

    Parameters
    ----------
    uplift : array-like of shape (n_samples,)
        Predicted treatment effect for each sample.
    treatment : array-like of shape (n_samples,)
        Binary treatment indicator.
    y : array-like of shape (n_samples,)
        Numeric outcome.
    normalize : bool, default=False
        If ``True``, divide the curve by its final value so it ends at 1.

    Returns
    -------
    x : np.ndarray of shape (n_samples + 1,)
        Fraction of the population treated, from 0 to 1.
    curve : np.ndarray of shape (n_samples + 1,)
        Cumulative incremental gain at each prefix.

    Raises
    ------
    ValueError
        If ``treatment`` is not binary or is missing one of the two groups.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.causal import qini_curve
    >>> rng = np.random.RandomState(0)
    >>> treatment = rng.randint(0, 2, size=200)
    >>> y = treatment * 2.0 + rng.normal(0, 0.1, size=200)
    >>> uplift = y  # a perfect ranking correlates with the outcome
    >>> x, curve = qini_curve(uplift, treatment, y)
    >>> x.shape
    (201,)
    """
    t, y_sorted = _coerce_rank_inputs(uplift, treatment, y)
    n = t.size
    n_t = float(np.sum(t))
    n_c = float(n - n_t)

    cum_yt = np.cumsum(y_sorted * t)
    cum_yc = np.cumsum(y_sorted * (1.0 - t))

    ratio = n_t / n_c if n_c > 0 else 0.0
    gain = cum_yt - ratio * cum_yc

    x = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    curve = np.concatenate([[0.0], gain])

    if normalize:
        last = curve[-1]
        if last != 0.0:
            curve = curve / last
    return x, curve


def auuc(uplift, treatment, y) -> float:
    """Area under the uplift curve (Qini coefficient).

    Computes the area under the Qini curve minus the area under the diagonal
    that a *random* ranking would trace. A value above zero means the model
    ranks high-uplift individuals ahead of low-uplift ones better than chance.

    Parameters
    ----------
    uplift : array-like of shape (n_samples,)
        Predicted treatment effect for each sample.
    treatment : array-like of shape (n_samples,)
        Binary treatment indicator.
    y : array-like of shape (n_samples,)
        Numeric outcome.

    Returns
    -------
    auuc : float
        The Qini coefficient (area under the curve minus the random diagonal).

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.causal import auuc
    >>> rng = np.random.RandomState(0)
    >>> treatment = rng.randint(0, 2, size=400)
    >>> y = treatment * 2.0 + rng.normal(0, 0.1, size=400)
    >>> good = auuc(y, treatment, y)
    >>> bad = auuc(rng.normal(0, 1, size=400), treatment, y)
    >>> bool(good > bad)
    True
    """
    x, curve = qini_curve(uplift, treatment, y)
    area = float(np.trapezoid(curve, x))
    # A random ranking would trace a straight line from (0, 0) to (1, curve[-1]).
    random_area = 0.5 * float(curve[-1])
    return area - random_area


def uplift_at_k(uplift, treatment, y, k: int = 100) -> float:
    """Mean uplift of the top-``k`` predicted-treatment-effect group.

    Ranks samples by predicted uplift and returns the difference between the
    mean outcome of the treated and control samples among the top ``k``.

    Parameters
    ----------
    uplift : array-like of shape (n_samples,)
        Predicted treatment effect for each sample.
    treatment : array-like of shape (n_samples,)
        Binary treatment indicator.
    y : array-like of shape (n_samples,)
        Numeric outcome.
    k : int, default=100
        Number of top-ranked samples to consider.

    Returns
    -------
    uplift : float
        ``mean(y | treated, top-k) - mean(y | control, top-k)``.

    Raises
    ------
    ValueError
        If the top-``k`` group does not contain both treatment groups.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.causal import uplift_at_k
    >>> treatment = np.tile([0, 1], 200)   # balanced groups
    >>> y = treatment * 2.0                # treated outcome 2, control 0
    >>> uplift = np.arange(400)            # top-k = indices 300..399 (mixed)
    >>> round(uplift_at_k(uplift, treatment, y, k=100), 1)
    2.0
    """
    t, y_sorted = _coerce_rank_inputs(uplift, treatment, y)
    n = t.size
    if k > n:
        k = n
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")

    top_t = t[:k]
    top_y = y_sorted[:k]
    mask_treated = top_t == 1
    if not np.any(mask_treated) or not np.any(~mask_treated):
        raise ValueError(
            "top-k group must contain both treated and control samples; "
            f"got {int(np.sum(mask_treated))} treated and "
            f"{int(np.sum(~mask_treated))} control"
        )
    return float(np.mean(top_y[mask_treated])) - float(np.mean(top_y[~mask_treated]))
