"""Global model explanations: a surrogate tree and feature-interaction strength."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from tuiml.explain._base import Explanation

__all__ = ["surrogate_tree", "friedman_h_statistic"]


def _model_output(estimator: Any, X: np.ndarray) -> np.ndarray:
    """Reduce a model's output to one number per sample.

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
        value for a regressor, max-probability otherwise.
    """
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X), dtype=np.float64)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        if proba.ndim == 2:
            return proba.max(axis=1)
    return np.asarray(estimator.predict(X), dtype=np.float64)


def surrogate_tree(
    estimator: Any,
    X: np.ndarray,
    max_depth: int = 3,
    random_state: Optional[int] = None,
) -> Explanation:
    """Approximate a model with a single **readable decision tree**.

    An ensemble or a neural network cannot be read by a person. A shallow tree
    can. The surrogate trick is to train one on the *model's own predictions*:
    the tree learns not the true labels but the pattern of "what the black box
    says", so its splits describe the model's behaviour, which is the thing
    being explained. Because the surrogate is a genuine TuiML estimator, every
    tool that works on a tree — :class:`~tuiml.explain.TreeExplainer`,
    plotting, :func:`~tuiml.explain.partial_dependence` — then works on the
    explanation.

    Overview
    --------
    1. Collect the model's predictions on ``X``.
    2. Fit a decision tree to those predictions, keeping the depth small.
    3. Return the fitted tree; its accuracy on the predictions is the
       ``fidelity``.

    Parameters
    ----------
    estimator : Algorithm
        A fitted model, of any kind.
    X : np.ndarray of shape (n_samples, n_features)
        Data to trace the model over. Use a held-out set or a representative
        sample; the surrogate only explains behaviour the data exercises.
    max_depth : int, default=3
        Depth of the surrogate. Three keeps it readable; deeper captures more
        of the model at the cost of unreadability, which defeats the purpose.
    random_state : int, optional
        Seed for the tree.

    Returns
    -------
    explanation : Explanation
        ``values`` holds nothing — the payload is ``metadata['tree']``, the
        fitted surrogate. ``metadata['fidelity']`` is the surrogate's R² or
        accuracy on the model's predictions.

    Notes
    -----
    **Fidelity is the number that decides whether to trust it.** A surrogate
    that reproduces the model's predictions faithfully is a fair summary of
    the model; one that does not is a fair summary of nothing. There is no
    universal threshold — check it against a plain baseline — but a surrogate
    barely better than guessing says the model has no shallow structure, which
    is itself a useful thing to learn.

    The surrogate describes **global** behaviour where the data is dense and
    nothing at all where it is sparse. It will also, by construction, miss
    detail the shallow tree cannot represent — that is the price of
    readability, and it is exactly the trade :class:`~tuiml.explain.TreeExplainer`
    avoids by being per-sample and exact.

    References
    ----------
    .. [Craven1996] Craven, M., & Shavlik, J. W. (1996). Extracting
       Tree-Structured Representations of Trained Networks. *NeurIPS*,
       24-30.
    .. [Ribeiro2016] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why
       Should I Trust You?": Explaining the Predictions of Any Classifier.
       *ACM SIGKDD*, 1135-1144. :doi:`10.1145/2939672.2939778`

    See Also
    --------
    :class:`~tuiml.explain.TreeExplainer` : Exact per-sample attributions for tree models.
    :func:`~tuiml.explain.friedman_h_statistic` : Quantifies a specific interaction, rather than summarising the whole model.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.explain import surrogate_tree
    >>> from tuiml.algorithms.trees import RandomForestRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 4))
    >>> y = 3.0 * X[:, 0] - X[:, 1] + rng.normal(0, 0.2, 400)
    >>> model = RandomForestRegressor(n_estimators=60, random_state=0).fit(X, y)
    >>> result = surrogate_tree(model, X, max_depth=3, random_state=0)
    >>> result.metadata['fidelity'] > 0.8
    True
    >>> result.metadata['tree'].tree_ is not None
    True
    """
    from tuiml.algorithms.trees import DecisionTreeRegressor

    X = np.asarray(X, dtype=np.float64)
    targets = _model_output(estimator, X)

    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    tree.fit(X, targets)

    from tuiml.evaluation.metrics import r2_score

    fidelity = float(r2_score(targets, tree.predict(X)))

    return Explanation(
        values=np.asarray([]),
        method="surrogate_tree",
        metadata={"tree": tree, "fidelity": fidelity},
    )


def friedman_h_statistic(
    estimator: Any,
    X: np.ndarray,
    feature_a: int,
    feature_b: int,
    n_points: int = 20,
    random_state: Optional[int] = None,
) -> float:
    """Quantify how much two features **interact**, in the Friedman sense.

    Most attribution methods attribute each feature its own share. An
    interaction is what is left after both have been credited — the part of
    the prediction that depends on the *combination*, in a way neither feature
    explains alone. Friedman's H-statistic measures that residual directly:
    it is 0 for a purely additive model and grows as the interaction does.

    Overview
    --------
    1. Compute the two single-feature partial-dependence functions, which
       capture what each feature explains on its own.
    2. Compute the joint partial-dependence surface, which captures both
       together.
    3. :math:`H^2` is the variance in the joint surface left unexplained by the
       sum of the two single ones.

    Theory
    ------
    With :math:`PD_{jk}` the joint partial dependence and :math:`PD_j`,
    :math:`PD_k` the individual ones,

    .. math::
        H_{jk}^2 = \\frac{
        \\sum_i \\left( \\tilde{PD}_{jk} - \\tilde{PD}_j - \\tilde{PD}_k \\right)^2
        }{
        \\sum_i \\tilde{PD}_{jk}^{\\,2}
        }

    where the tilde marks centred partial dependences, each shifted so its mean
    over the evaluation grid is zero. Centring matters: an uncentred model
    with a constant level :math:`c` would report an interaction of
    :math:`H^2 = 1` between two features it does not use, because the flat
    joint surface :math:`c` compared against two flat curves summing to
    :math:`2c` leaves a residual of :math:`-c`.

    evaluated over the grid of observed values. A model with no interaction
    has :math:`PD_{jk} = PD_j + PD_k` exactly, so :math:`H^2 = 0`; the more of
    the surface the two single curves fail to explain, the closer it gets to 1.

    Parameters
    ----------
    estimator : Algorithm
        A fitted model.
    X : np.ndarray of shape (n_samples, n_features)
        Data defining the grid and the rows averaged over.
    feature_a, feature_b : int
        The pair to test.
    n_points : int, default=20
        Grid points per feature. Cost grows with its square.
    random_state : int, optional
        Seed for downsampling large grids.

    Returns
    -------
    h_squared : float
        Interaction strength in ``[0, 1]``. Roughly: below 0.01 negligible,
        above 0.1 worth investigating. Values above 1 are possible when the
        single curves happen to anti-correlate with the joint surface; treat
        them as "strong", not as a violation.

    Notes
    -----
    **Complexity.** :math:`O(n^2_{\\text{points}})` prediction passes over
    ``X``, so keep ``n_points`` modest — 20 gives 400 passes.

    **A low value is an assumption confirmed, not a null result.** H measures
    the interaction *on the grid of observed values*; an interaction confined
    to a region the data never visits is invisible to it. And like
    :func:`~tuiml.explain.partial_dependence`, the marginalisation can
    evaluate the model on feature combinations the data does not contain.

    References
    ----------
    .. [Friedman2008] Friedman, J. H., & Popescu, B. E. (2008). Predictive
       Learning via Rule Ensembles. *Annals of Applied Statistics*, 2(3),
       916-954. :doi:`10.1214/07-AOAS148`

    See Also
    --------
    :func:`~tuiml.explain.partial_dependence` : The single-feature curves this method extends.
    :func:`~tuiml.explain.surrogate_tree` : The whole-model counterpart.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.explain import friedman_h_statistic
    >>> from tuiml.algorithms.trees import RandomForestRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 3))
    >>> additive = 3.0 * X[:, 0] + 2.0 * X[:, 1] + rng.normal(0, 0.2, 400)
    >>> interaction = 3.0 * X[:, 0] * X[:, 1] + rng.normal(0, 0.2, 400)
    >>> add = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, additive)
    >>> inter = RandomForestRegressor(n_estimators=40, random_state=0).fit(X, interaction)
    >>> h_add = friedman_h_statistic(add, X, 0, 1, n_points=8, random_state=0)
    >>> h_inter = friedman_h_statistic(inter, X, 0, 1, n_points=8, random_state=0)
    >>> bool(h_inter > h_add)
    True
    """
    from tuiml.explain.dependence import _grid

    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(random_state)

    grid_a = _grid(X[:, feature_a], n_points)
    grid_b = _grid(X[:, feature_b], n_points)

    def single(feature: int, grid: np.ndarray) -> np.ndarray:
        """Return the partial-dependence curve for one feature."""
        out = np.empty(len(grid))
        for column, point in enumerate(grid):
            altered = X.copy()
            altered[:, feature] = point
            out[column] = _model_output(estimator, altered).mean()
        return out

    curve_a = single(feature_a, grid_a)
    curve_b = single(feature_b, grid_b)

    # The joint surface, marginalising everything but the two features.
    joint = np.empty((len(grid_a), len(grid_b)))
    for i, point_a in enumerate(grid_a):
        for j, point_b in enumerate(grid_b):
            altered = X.copy()
            altered[:, feature_a] = point_a
            altered[:, feature_b] = point_b
            joint[i, j] = _model_output(estimator, altered).mean()

    # Friedman's definition is over *centred* partial dependences: each is
    # shifted so its mean over the grid is zero. Without the shift, the
    # constant level of the model appears as a spurious interaction — a flat
    # joint surface of height c would leave a residual of -c against two flat
    # curves of height c, reporting a large interaction where there is none.
    joint_c = joint - joint.mean()
    curve_a_c = curve_a - curve_a.mean()
    curve_b_c = curve_b - curve_b.mean()

    # A flat joint surface — the model ignores both features — makes the
    # statistic 0/0. Floating point leaves a ~1e-17 residue in the centred
    # surface that would resolve that ratio to 1.0, reporting perfect
    # interaction where there is none. Treat a constant surface as no
    # interaction instead.
    joint_scale = float(np.abs(joint).max())
    if joint_scale <= 0 or (joint.max() - joint.min()) <= 1e-12 * joint_scale:
        return 0.0

    additive = curve_a_c[:, None] + curve_b_c[None, :]
    residual = joint_c - additive

    numerator = float((residual ** 2).sum())
    denominator = float((joint_c ** 2).sum())
    return numerator / denominator if denominator > 0 else 0.0
