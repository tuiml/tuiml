"""Partial dependence, ICE and ALE - how the prediction moves with a feature."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from tuiml.explain._base import Explanation

__all__ = [
    "partial_dependence",
    "individual_conditional_expectation",
    "accumulated_local_effects",
]


def _model_output(estimator: Any, X: np.ndarray) -> np.ndarray:
    """Return a one-dimensional model output suitable for averaging.

    Parameters
    ----------
    estimator : Algorithm
        A fitted model.
    X : np.ndarray of shape (n_samples, n_features)
        Inputs.

    Returns
    -------
    output : np.ndarray of shape (n_samples,)
        Positive-class probability for a binary classifier, the predicted
        value for a regressor, and the max-probability otherwise.
    """
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X), dtype=np.float64)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        if proba.ndim == 2:
            return proba.max(axis=1)
    return np.asarray(estimator.predict(X), dtype=np.float64)


def _grid(values: np.ndarray, n_points: int) -> np.ndarray:
    """Build an evaluation grid over a feature's observed range.

    Quantiles rather than equal spacing, so the grid follows the data instead
    of stretching across empty regions a single outlier opened up.

    Parameters
    ----------
    values : np.ndarray of shape (n_samples,)
        Observed values of one feature.
    n_points : int
        Grid size requested.

    Returns
    -------
    grid : np.ndarray
        Sorted unique grid points, possibly fewer than ``n_points``.
    """
    unique = np.unique(values)
    if len(unique) <= n_points:
        return unique
    return np.unique(np.quantile(values, np.linspace(0, 1, n_points)))


def individual_conditional_expectation(
    estimator: Any,
    X: np.ndarray,
    feature: int,
    n_points: int = 30,
    feature_names: Optional[List[str]] = None,
) -> Explanation:
    """Trace **each sample's** prediction as one feature is swept.

    One curve per row: hold everything else fixed, vary the feature across a
    grid, and record what the model says. Averaging these curves gives partial
    dependence — which is exactly why you should look at them first. A flat
    average can hide curves that rise for half the population and fall for the
    other half, and only the individual curves reveal it.

    Parameters
    ----------
    estimator : Algorithm
        A fitted model.
    X : np.ndarray of shape (n_samples, n_features)
        Data defining the population and the grid range.
    feature : int
        Index of the feature to sweep.
    n_points : int, default=30
        Grid points, placed at quantiles of the observed values.
    feature_names : list of str, optional
        Names for the report.

    Returns
    -------
    explanation : Explanation
        ``values`` has shape ``(n_samples, n_grid)``. ``metadata['grid']``
        holds the grid, ``metadata['average']`` the partial-dependence curve.

    Notes
    -----
    **Complexity.** ``n_grid`` prediction passes over the whole of ``X``.

    Like partial dependence, this evaluates the model on inputs the data may
    never contain — setting one feature to a value incompatible with the rest
    of the row. Read curves in dense regions with more confidence than in
    sparse ones.

    References
    ----------
    .. [Goldstein2015] Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E.
       (2015). Peeking Inside the Black Box: Visualizing Statistical Learning
       with Plots of Individual Conditional Expectation. *Journal of
       Computational and Graphical Statistics*, 24(1), 44-65.
       :doi:`10.1080/10618600.2014.907095`

    See Also
    --------
    :func:`~tuiml.explain.partial_dependence` : The average of these curves.
    :func:`~tuiml.explain.accumulated_local_effects` : Avoids evaluating impossible inputs.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.explain import individual_conditional_expectation
    >>> from tuiml.algorithms.trees import RandomForestRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(120, 3))
    >>> y = 2.0 * X[:, 0] + rng.normal(0, 0.1, 120)
    >>> model = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y)
    >>> ice = individual_conditional_expectation(model, X, feature=0, n_points=10)
    >>> ice.values.shape[0]
    120
    """
    X = np.asarray(X, dtype=np.float64)
    grid = _grid(X[:, feature], n_points)

    curves = np.empty((len(X), len(grid)))
    for column, point in enumerate(grid):
        altered = X.copy()
        altered[:, feature] = point
        curves[:, column] = _model_output(estimator, altered)

    return Explanation(
        values=curves,
        feature_names=feature_names,
        method="individual_conditional_expectation",
        metadata={
            "grid": grid,
            "feature": feature,
            "average": curves.mean(axis=0),
        },
    )


def partial_dependence(
    estimator: Any,
    X: np.ndarray,
    feature: int,
    n_points: int = 30,
    feature_names: Optional[List[str]] = None,
) -> Explanation:
    """The **average** effect of a feature on the prediction.

    Sweeps a feature across a grid, marginalising the others by averaging over
    the observed data, and reports the mean prediction at each point. It
    answers "as this feature rises, what happens on average?" — the single
    most-asked question about a fitted model.

    Parameters
    ----------
    estimator : Algorithm
        A fitted model.
    X : np.ndarray of shape (n_samples, n_features)
        Data used to marginalise the other features.
    feature : int
        Index of the feature to sweep.
    n_points : int, default=30
        Grid points, placed at quantiles of the observed values.
    feature_names : list of str, optional
        Names for the report.

    Returns
    -------
    explanation : Explanation
        ``values`` is the curve, of shape ``(n_grid,)``.
        ``metadata['grid']`` holds the grid and ``metadata['ice']`` the
        individual curves it averages.

    Notes
    -----
    **Complexity.** ``n_grid`` prediction passes over the whole of ``X``.

    **Two failure modes worth knowing before trusting a flat curve.** First,
    averaging hides heterogeneity: curves rising for half the population and
    falling for the other half average to nothing. Always glance at
    :func:`individual_conditional_expectation` before concluding a feature
    does not matter. Second, marginalising by averaging over the data creates
    rows the world cannot produce — a house with two rooms and 400 square
    metres — and the model is scored on them anyway. Where features are
    strongly correlated, :func:`accumulated_local_effects` avoids that by
    only ever perturbing within a local window.

    References
    ----------
    .. [Friedman2001] Friedman, J. H. (2001). Greedy Function Approximation:
       A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189-1232.
       :doi:`10.1214/aos/1013203451`

    See Also
    --------
    :func:`~tuiml.explain.individual_conditional_expectation` : The curves this averages.
    :func:`~tuiml.explain.accumulated_local_effects` : Correct under correlated features.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.explain import partial_dependence
    >>> from tuiml.algorithms.trees import RandomForestRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> y = 2.0 * X[:, 0] + rng.normal(0, 0.1, 200)
    >>> model = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y)
    >>> pd = partial_dependence(model, X, feature=0, n_points=12)
    >>> bool(np.all(np.diff(pd.values) > -0.05))     # rises with the feature
    True
    """
    ice = individual_conditional_expectation(
        estimator, X, feature, n_points=n_points, feature_names=feature_names
    )
    return Explanation(
        values=ice.metadata["average"],
        feature_names=feature_names,
        method="partial_dependence",
        metadata={
            "grid": ice.metadata["grid"],
            "feature": feature,
            "ice": ice.values,
        },
    )


def accumulated_local_effects(
    estimator: Any,
    X: np.ndarray,
    feature: int,
    n_bins: int = 20,
    feature_names: Optional[List[str]] = None,
) -> Explanation:
    """A feature's effect, computed **without inventing impossible rows**.

    Partial dependence asks what the model says when a feature is set to a
    value the rest of the row makes implausible. ALE never does that: it
    divides the feature into quantile bins, and inside each bin asks only how
    the prediction changes between that bin's own edges, for the rows that
    actually fall there. Those local differences are then accumulated into a
    curve.

    Under correlated features this is the difference between a trustworthy
    curve and an artefact.

    Overview
    --------
    1. Cut the feature into quantile bins.
    2. For rows in a bin, predict twice — once with the feature at the bin's
       lower edge, once at its upper edge.
    3. Average that difference within the bin: the **local effect**.
    4. Accumulate across bins, then centre so the curve averages to zero.

    Theory
    ------
    With bins :math:`(z_{k-1}, z_k]` the uncentred ALE is

    .. math::
        \\widehat{\\mathrm{ALE}}(x) = \\sum_{k=1}^{k(x)}
        \\frac{1}{n_k} \\sum_{i : x_i \\in \\text{bin } k}
        \\left[ f(z_k, x_{i,\\setminus j}) - f(z_{k-1}, x_{i,\\setminus j}) \\right]

    and is then centred by subtracting its data-weighted mean. Every
    evaluation keeps a row's other features intact and moves the target
    feature only within the narrow band it already occupies, so the model is
    never asked about a combination the data does not support.

    Parameters
    ----------
    estimator : Algorithm
        A fitted model.
    X : np.ndarray of shape (n_samples, n_features)
        Data defining the bins and the rows evaluated in each.
    feature : int
        Index of the feature to analyse.
    n_bins : int, default=20
        Quantile bins. More bins give finer resolution and noisier local
        effects, since each holds fewer rows.
    feature_names : list of str, optional
        Names for the report.

    Returns
    -------
    explanation : Explanation
        ``values`` is the centred ALE curve at each bin edge.
        ``metadata['grid']`` holds the edges and ``metadata['bin_counts']``
        the rows behind each estimate.

    Notes
    -----
    **Complexity.** Two prediction passes over the data in total, regardless
    of bin count — cheaper than partial dependence, which costs one pass per
    grid point.

    **The curve is an effect, not a level.** ALE is centred at zero and says
    how the prediction changes *relative to the average*, so its absolute
    height carries no meaning; only its shape and range do.

    Bins holding few rows give noisy local effects. Check
    ``metadata['bin_counts']`` before reading anything into a sharp move at
    the tails.

    References
    ----------
    .. [Apley2020] Apley, D. W., & Zhu, J. (2020). Visualizing the Effects of
       Predictor Variables in Black Box Supervised Learning Models. *Journal
       of the Royal Statistical Society: Series B*, 82(4), 1059-1086.
       :doi:`10.1111/rssb.12377`

    See Also
    --------
    :func:`~tuiml.explain.partial_dependence` : Simpler, but unreliable under correlation.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.explain import accumulated_local_effects
    >>> from tuiml.algorithms.trees import RandomForestRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 2))
    >>> X[:, 1] = X[:, 0] + rng.normal(0, 0.1, 300)     # strongly correlated
    >>> y = 2.0 * X[:, 0] + rng.normal(0, 0.1, 300)
    >>> model = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y)
    >>> ale = accumulated_local_effects(model, X, feature=0, n_bins=10)
    >>> bool(ale.values[-1] > ale.values[0])           # increasing effect
    True
    """
    X = np.asarray(X, dtype=np.float64)
    values = X[:, feature]

    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        # A constant feature has no effect to accumulate.
        return Explanation(
            values=np.zeros(1),
            feature_names=feature_names,
            method="accumulated_local_effects",
            metadata={"grid": edges, "bin_counts": np.zeros(1), "feature": feature},
        )

    # Bin index per row, clipped so the maximum value joins the last bin.
    index = np.clip(np.searchsorted(edges, values, side="left") - 1, 0, len(edges) - 2)

    lower = X.copy()
    upper = X.copy()
    lower[:, feature] = edges[index]
    upper[:, feature] = edges[index + 1]
    difference = _model_output(estimator, upper) - _model_output(estimator, lower)

    n_intervals = len(edges) - 1
    effects = np.zeros(n_intervals)
    counts = np.zeros(n_intervals)
    for bin_index in range(n_intervals):
        member = index == bin_index
        counts[bin_index] = member.sum()
        if counts[bin_index] > 0:
            effects[bin_index] = difference[member].mean()

    curve = np.concatenate([[0.0], np.cumsum(effects)])

    # Centre by the data-weighted mean, so the curve reads as a deviation from
    # the average prediction rather than an arbitrary offset.
    midpoints = (curve[:-1] + curve[1:]) / 2.0
    total = counts.sum()
    centre = float((midpoints * counts).sum() / total) if total > 0 else 0.0

    return Explanation(
        values=curve - centre,
        feature_names=feature_names,
        method="accumulated_local_effects",
        metadata={"grid": edges, "bin_counts": counts, "feature": feature},
    )
