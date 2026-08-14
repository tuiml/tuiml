"""Elastic distance measures for time series."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from tuiml._cpp_ext import timeseries as _cpp_ts
from tuiml.algorithms.timeseries.classification._base import as_panel

__all__ = ["dtw_distance", "dtw_pairwise", "lb_keogh", "lb_keogh_envelope"]


def _resolve_window(window: Optional[float], n_timepoints: int) -> int:
    """Convert a window given as a fraction or a count into time steps.

    Parameters
    ----------
    window : float, int or None
        ``None`` for no constraint, a float in ``(0, 1]`` read as a fraction of
        the series length, or an int read as a step count.
    n_timepoints : int
        Series length, used to resolve a fractional window.

    Returns
    -------
    steps : int
        Band half-width in time steps; ``-1`` means unconstrained.
    """
    if window is None:
        return -1
    if isinstance(window, float) and 0.0 < window <= 1.0:
        return max(1, int(round(window * n_timepoints)))
    return int(window)


def dtw_distance(
    a: np.ndarray, b: np.ndarray, window: Optional[float] = None
) -> float:
    """Dynamic Time Warping distance between two series.

    DTW aligns two series by **stretching and compressing the time axis**, so
    two signals with the same shape but different timing compare as similar
    where Euclidean distance would call them far apart.

    Parameters
    ----------
    a : np.ndarray of shape (n_timepoints,)
        First series.
    b : np.ndarray of shape (n_timepoints,)
        Second series. Lengths need not match.
    window : float or int, optional
        Sakoe-Chiba band half-width. A float in ``(0, 1]`` is a fraction of the
        series length; an int is a step count; ``None`` leaves the warping
        unconstrained. A band both speeds the computation up and usually
        **improves** accuracy, by forbidding pathological alignments that map
        one point onto half the other series.

    Returns
    -------
    distance : float
        DTW distance. Zero only when the series are identical.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.classification import dtw_distance
    >>> a = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    >>> shifted = np.array([0.0, 0.0, 1.0, 2.0, 1.0])
    >>> float(dtw_distance(a, a))
    0.0
    >>> bool(dtw_distance(a, shifted) < np.linalg.norm(a - shifted))
    True
    """
    a = np.ascontiguousarray(a, dtype=np.float64).ravel()
    b = np.ascontiguousarray(b, dtype=np.float64).ravel()
    return float(
        _cpp_ts.dtw_distance(a, b, _resolve_window(window, max(a.size, b.size)), 0.0)
    )


def dtw_pairwise(
    A: np.ndarray, B: Optional[np.ndarray] = None, window: Optional[float] = None
) -> np.ndarray:
    """Full DTW distance matrix between two panels of series.

    Parameters
    ----------
    A : np.ndarray of shape (n_a, n_timepoints) or (n_a, n_channels, n_timepoints)
        First panel.
    B : np.ndarray, optional
        Second panel. Defaults to ``A``, giving the self-distance matrix.
    window : float or int, optional
        Sakoe-Chiba band half-width; see :func:`dtw_distance`.

    Returns
    -------
    distances : np.ndarray of shape (n_a, n_b)
        Pairwise DTW distances.

    Notes
    -----
    Multivariate panels use **dependent** DTW: one warping path is shared
    across channels, with the per-channel differences summed into the local
    cost. That is the right default when channels are synchronised readings of
    one process, and the wrong one when they warp independently.

    Cost is :math:`O(n_a n_b L^2)` before the band, so prefer
    :class:`~tuiml.algorithms.timeseries.classification.DTWNeighborsClassifier`
    when only the nearest neighbours are needed — it prunes with LB_Keogh and
    never builds the full matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.classification import dtw_pairwise
    >>> panel = np.random.default_rng(0).normal(size=(4, 30))
    >>> distances = dtw_pairwise(panel, window=0.1)
    >>> distances.shape
    (4, 4)
    >>> bool(np.allclose(np.diag(distances), 0.0))
    True
    """
    panel_a = as_panel(A)
    panel_b = panel_a if B is None else as_panel(B)
    steps = _resolve_window(window, panel_a.shape[2])
    return np.asarray(_cpp_ts.dtw_pairwise(panel_a, panel_b, steps))


def lb_keogh_envelope(
    series: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Running min/max envelope of a series under a Sakoe-Chiba band.

    Parameters
    ----------
    series : np.ndarray of shape (n_timepoints,)
        Input series.
    window : int
        Band half-width in time steps.

    Returns
    -------
    lower : np.ndarray of shape (n_timepoints,)
        Running minimum over each window.
    upper : np.ndarray of shape (n_timepoints,)
        Running maximum over each window.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.classification import lb_keogh_envelope
    >>> lower, upper = lb_keogh_envelope(np.array([1.0, 5.0, 2.0, 4.0]), 1)
    >>> bool(np.all(lower <= upper))
    True
    """
    series = np.ascontiguousarray(series, dtype=np.float64).ravel()
    lower, upper = _cpp_ts.lb_keogh_envelope(series, int(window))
    return np.asarray(lower), np.asarray(upper)


def lb_keogh(
    query: np.ndarray, reference: np.ndarray, window: int
) -> float:
    """LB_Keogh lower bound of the DTW distance.

    A cheap :math:`O(n)` quantity that is **guaranteed never to exceed** the
    true DTW distance. That guarantee is what makes it useful: during a
    nearest-neighbour search, a candidate whose bound already loses to the
    current best cannot win, so its DTW never has to be computed.

    Parameters
    ----------
    query : np.ndarray of shape (n_timepoints,)
        Query series.
    reference : np.ndarray of shape (n_timepoints,)
        Reference series, whose envelope the query is compared against.
    window : int
        Band half-width in time steps.

    Returns
    -------
    bound : float
        Lower bound on ``dtw_distance(query, reference, window)``.

    Notes
    -----
    The bound is asymmetric — swapping the arguments gives a different, also
    valid, bound. Implementations that need the tightest available value take
    the maximum of both directions.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.timeseries.classification import dtw_distance, lb_keogh
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.normal(size=50), rng.normal(size=50)
    >>> bool(lb_keogh(a, b, 5) <= dtw_distance(a, b, 5))
    True
    """
    query = np.ascontiguousarray(query, dtype=np.float64).ravel()
    reference = np.ascontiguousarray(reference, dtype=np.float64).ravel()
    lower, upper = _cpp_ts.lb_keogh_envelope(reference, int(window))
    return float(_cpp_ts.lb_keogh(query, lower, upper))
