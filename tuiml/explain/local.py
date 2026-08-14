"""Local, model-agnostic explanations: LIME surrogates and counterfactuals."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from tuiml.explain._base import Explanation

__all__ = ["lime_explain", "counterfactual"]


def _scalar_output(estimator: Any, X: np.ndarray, class_index: Optional[int]) -> np.ndarray:
    """Reduce a model's output to one number per sample.

    Parameters
    ----------
    estimator : Algorithm
        A fitted model.
    X : np.ndarray of shape (n_samples, n_features)
        Inputs.
    class_index : int, optional
        Which class probability to explain. Ignored for regressors.

    Returns
    -------
    output : np.ndarray of shape (n_samples,)
        Probability of the chosen class, or the predicted value.
    """
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X), dtype=np.float64)
        if proba.ndim == 2:
            column = 0 if class_index is None else int(class_index)
            return proba[:, min(column, proba.shape[1] - 1)]
    return np.asarray(estimator.predict(X), dtype=np.float64)


def lime_explain(
    estimator: Any,
    x: np.ndarray,
    background: np.ndarray,
    class_index: Optional[int] = None,
    n_samples: int = 2000,
    kernel_width: Optional[float] = None,
    n_features: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> Explanation:
    """Explain one prediction by fitting a **local linear model** around it.

    LIME's premise is that a model too complicated to describe globally is
    often nearly linear in a small neighbourhood. It perturbs the sample,
    asks the model what it thinks of each perturbation, weights those by how
    close they landed, and fits a weighted linear model. The coefficients are
    the explanation — and because only ``predict`` is ever called, this works
    on **any** model, including ones :class:`~tuiml.explain.TreeExplainer`
    cannot touch.

    Overview
    --------
    1. Draw perturbations by sampling each feature from the background's
       distribution.
    2. Query the model on every perturbation.
    3. Weight each by an exponential kernel on its distance from ``x``.
    4. Fit a weighted ridge regression; report its coefficients.

    Theory
    ------
    LIME solves

    .. math::
        \\xi(x) = \\arg\\min_{g \\in G} \\ \\mathcal{L}(f, g, \\pi_x)
        + \\Omega(g)

    over interpretable models :math:`g`, with locality supplied by the kernel

    .. math::
        \\pi_x(z) = \\exp\\left( -\\frac{d(x, z)^2}{\\sigma^2} \\right)

    Distances are measured in units of the background's standard deviation, so
    a feature measured in millions does not swamp one measured in units.

    Parameters
    ----------
    estimator : Algorithm
        A fitted model. Only ``predict`` or ``predict_proba`` is used.
    x : np.ndarray of shape (n_features,)
        The single sample to explain.
    background : np.ndarray of shape (n_background, n_features)
        Data defining the perturbation distribution and the feature scales.
    class_index : int, optional
        Which class probability to explain, for a classifier.
    n_samples : int, default=2000
        Perturbations drawn. More is steadier and slower.
    kernel_width : float, optional
        Locality width in standardised units. Defaults to
        :math:`0.75 \\sqrt{d}`, the usual heuristic.
    n_features : int, optional
        Report only this many coefficients, chosen by magnitude. ``None``
        keeps all.
    feature_names : list of str, optional
        Names for the report.
    random_state : int, optional
        Seed for the perturbations.

    Returns
    -------
    explanation : Explanation
        ``values`` holds one coefficient per feature. ``metadata`` carries
        ``intercept``, ``local_r2`` — the surrogate's weighted fit quality —
        and ``prediction``.

    Notes
    -----
    **Complexity.** One prediction pass over ``n_samples`` rows, plus a
    least-squares solve. Explaining a whole dataset means repeating that per
    row, which is why LIME is a tool for inspecting individual decisions
    rather than summarising a model.

    **Check ``local_r2`` before believing the coefficients.** They describe
    the *surrogate*, and if the surrogate fits badly the explanation describes
    nothing. A low value means the model is not locally linear at this point,
    and the honest response is to say so rather than to read the coefficients
    anyway.

    **LIME is not additive and its explanations are not stable.** Unlike
    :class:`~tuiml.explain.TreeExplainer`, coefficients carry no guarantee of
    summing to the prediction, and re-running with a different seed can shift
    them — markedly so when ``n_samples`` is small or the kernel is narrow. If
    the model is a TuiML tree or forest, prefer TreeExplainer: it is exact,
    deterministic and cheaper.

    Perturbing features independently also fabricates rows the data never
    contains, the same criticism that applies to
    :func:`~tuiml.explain.partial_dependence`.

    References
    ----------
    .. [Ribeiro2016] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why
       Should I Trust You?": Explaining the Predictions of Any Classifier.
       *ACM SIGKDD*, 1135-1144. :doi:`10.1145/2939672.2939778`

    See Also
    --------
    :class:`~tuiml.explain.TreeExplainer` : Exact and deterministic, for tree models.
    :func:`~tuiml.explain.counterfactual` : What would have to change, rather than what mattered.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.explain import lime_explain
    >>> from tuiml.algorithms.trees import RandomForestRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 4))
    >>> y = 5.0 * X[:, 2] + rng.normal(0, 0.1, 300)
    >>> model = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, y)
    >>> result = lime_explain(model, X[0], background=X, random_state=0)
    >>> result.top(1)[0][0]
    'feature_2'
    >>> bool(result.metadata['local_r2'] > 0.5)      # the surrogate fits
    True
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    background = np.asarray(background, dtype=np.float64)
    rng = np.random.default_rng(random_state)

    scale = background.std(axis=0)
    scale[scale <= 0] = 1.0
    n_dim = len(x)

    # Perturb in the background's own units, so every feature is disturbed by
    # a comparable amount whatever it is measured in.
    perturbed = rng.normal(loc=background.mean(axis=0), scale=scale,
                           size=(n_samples, n_dim))
    perturbed[0] = x  # keep the point itself in the sample

    outputs = _scalar_output(estimator, perturbed, class_index)

    if kernel_width is None:
        kernel_width = 0.75 * np.sqrt(n_dim)
    distance = np.sqrt((((perturbed - x) / scale) ** 2).sum(axis=1))
    weights = np.exp(-(distance ** 2) / (kernel_width ** 2))

    coefficients, intercept, r2 = _weighted_ridge(
        (perturbed - x) / scale, outputs, weights
    )
    # Report per original unit rather than per standard deviation.
    coefficients = coefficients / scale

    if n_features is not None and n_features < n_dim:
        keep = np.argsort(-np.abs(coefficients))[:n_features]
        mask = np.zeros(n_dim, dtype=bool)
        mask[keep] = True
        coefficients = np.where(mask, coefficients, 0.0)

    return Explanation(
        values=coefficients,
        feature_names=feature_names,
        method="lime",
        metadata={
            "intercept": intercept,
            "local_r2": r2,
            "prediction": float(_scalar_output(estimator, x[None, :], class_index)[0]),
            "kernel_width": float(kernel_width),
        },
    )


def _weighted_ridge(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float = 1e-3
) -> tuple:
    """Fit a weighted ridge regression and report its weighted fit quality.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Design matrix, already centred on the explained point.
    y : np.ndarray of shape (n_samples,)
        Model outputs to approximate.
    weights : np.ndarray of shape (n_samples,)
        Locality weights.
    alpha : float, default=1e-3
        Ridge penalty, keeping the solve stable when perturbations are nearly
        collinear.

    Returns
    -------
    coefficients : np.ndarray of shape (n_features,)
        Fitted slopes.
    intercept : float
        Fitted intercept.
    r2 : float
        Weighted coefficient of determination of the surrogate.
    """
    design = np.column_stack([np.ones(len(X)), X])
    root = np.sqrt(weights)[:, None]

    # Solve the weighted normal equations with a ridge term, leaving the
    # intercept unpenalised so it can absorb the local level.
    lhs = (design * root).T @ (design * root)
    penalty = alpha * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    rhs = (design * root).T @ (y * np.sqrt(weights))
    solution = np.linalg.solve(lhs + penalty, rhs)

    fitted = design @ solution
    total_weight = weights.sum()
    mean = float((weights * y).sum() / total_weight)
    residual = float((weights * (y - fitted) ** 2).sum())
    variance = float((weights * (y - mean) ** 2).sum())
    r2 = 1.0 - residual / variance if variance > 0 else 0.0

    return solution[1:], float(solution[0]), r2


def counterfactual(
    estimator: Any,
    x: np.ndarray,
    background: np.ndarray,
    target: Any,
    feature_names: Optional[List[str]] = None,
    max_features_changed: Optional[int] = None,
) -> Explanation:
    """Find the **smallest change** that would flip the prediction.

    Importance tells you what mattered. A counterfactual tells you what to
    *do*: "this loan would have been approved with £4,000 more income." That
    is the form an explanation has to take when someone is entitled to act on
    it, and it is the form regulation increasingly asks for.

    Overview
    --------
    1. Find the nearest background sample the model already assigns to the
       target class — a real, achievable point rather than an invented one.
    2. Walk each of its differing features back towards ``x``, keeping a
       change only while the prediction stays at the target.
    3. What survives is a sparse, minimal set of changes.

    Anchoring on a real observation is deliberate. Optimising freely in
    feature space produces mathematically minimal counterfactuals that are
    physically impossible — negative ages, a house with two rooms and 400
    square metres. Starting from data the world actually produced keeps the
    answer plausible.

    Parameters
    ----------
    estimator : Algorithm
        A fitted classifier exposing ``predict``.
    x : np.ndarray of shape (n_features,)
        The sample whose prediction should change.
    background : np.ndarray of shape (n_background, n_features)
        Pool of realistic points to search.
    target : Any
        Desired predicted label.
    feature_names : list of str, optional
        Names for the report.
    max_features_changed : int, optional
        Keep at most this many changes, largest first. Fewer changes are
        easier to act on, at the cost of a larger move in each.

    Returns
    -------
    explanation : Explanation
        ``values`` holds the change required per feature, zero where nothing
        need change. ``metadata`` carries the full ``counterfactual`` row,
        ``n_changed``, ``distance`` and ``found``.

    Notes
    -----
    **Complexity.** One prediction pass over the background, then at most one
    per differing feature.

    **A counterfactual is not advice.** It says the *model* would decide
    differently, not that acting on it would change the real outcome — those
    coincide only if the model is causal, which it generally is not. It also
    describes one route among many; a different search returns a different,
    equally valid answer.

    ``found`` is False when no background sample is classified as the target,
    which usually means the class is absent from the background rather than
    that no counterfactual exists.

    References
    ----------
    .. [Wachter2018] Wachter, S., Mittelstadt, B., & Russell, C. (2018).
       Counterfactual Explanations without Opening the Black Box.
       *Harvard Journal of Law & Technology*, 31(2), 841-887.
       :doi:`10.2139/ssrn.3063289`
    .. [Poyiadzi2020] Poyiadzi, R., Sokol, K., Santos-Rodriguez, R., De Bie,
       T., & Flach, P. (2020). FACE: Feasible and Actionable Counterfactual
       Explanations. *AAAI/ACM AIES*, 344-350. :doi:`10.1145/3375627.3375850`

    See Also
    --------
    :func:`~tuiml.explain.lime_explain` : What mattered, rather than what to change.
    :class:`~tuiml.explain.TreeExplainer` : Exact attributions for tree models.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.explain import counterfactual
    >>> from tuiml.algorithms.trees import DecisionTreeClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 3))
    >>> y = (X[:, 0] > 0).astype(int)          # only feature 0 decides
    >>> model = DecisionTreeClassifier(max_depth=4).fit(X, y)
    >>> negative = X[model.predict(X) == 0][0]
    >>> result = counterfactual(model, negative, background=X, target=1)
    >>> bool(result.metadata['found'])
    True
    >>> int(result.metadata['n_changed'])      # one feature suffices
    1
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    background = np.asarray(background, dtype=np.float64)

    predictions = np.asarray(estimator.predict(background))
    matches = np.flatnonzero(predictions == target)

    if matches.size == 0:
        return Explanation(
            values=np.zeros_like(x),
            feature_names=feature_names,
            method="counterfactual",
            metadata={
                "found": False,
                "n_changed": 0,
                "distance": float("inf"),
                "counterfactual": None,
            },
        )

    # Scale distances by feature spread so no single wide-ranging column
    # decides which candidate counts as nearest.
    scale = background.std(axis=0)
    scale[scale <= 0] = 1.0
    distances = np.sqrt(((background[matches] - x) / scale) ** 2).sum(axis=1)
    candidate = background[matches[int(np.argmin(distances))]].copy()

    # Revert features towards x wherever the target prediction survives,
    # which is what turns a nearest neighbour into a sparse counterfactual.
    order = np.argsort(np.abs((candidate - x) / scale))
    for column in order:
        if candidate[column] == x[column]:
            continue
        trial = candidate.copy()
        trial[column] = x[column]
        if estimator.predict(trial[None, :])[0] == target:
            candidate = trial

    delta = candidate - x
    changed = np.flatnonzero(delta != 0.0)

    if max_features_changed is not None and len(changed) > max_features_changed:
        keep = changed[
            np.argsort(-np.abs(delta[changed] / scale[changed]))[:max_features_changed]
        ]
        trimmed = x.copy()
        trimmed[keep] = candidate[keep]
        # Only accept the trimmed version if it still flips the prediction.
        if estimator.predict(trimmed[None, :])[0] == target:
            candidate = trimmed
            delta = candidate - x
            changed = np.flatnonzero(delta != 0.0)

    return Explanation(
        values=delta,
        feature_names=feature_names,
        method="counterfactual",
        metadata={
            "found": bool(estimator.predict(candidate[None, :])[0] == target),
            "n_changed": int(len(changed)),
            "distance": float(np.sqrt(((delta / scale) ** 2).sum())),
            "counterfactual": candidate,
            "target": target,
        },
    )
