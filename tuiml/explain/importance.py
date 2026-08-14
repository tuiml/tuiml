"""Model-agnostic feature importance by measuring what breaking a feature costs."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

import numpy as np

from tuiml.explain._base import Explanation


def _resolve_scorer(scoring: Union[str, Callable]) -> Callable:
    """Return a ``(y_true, y_pred) -> float`` scorer, higher being better.

    Parameters
    ----------
    scoring : str or callable
        ``'accuracy'``, ``'r2'``, ``'neg_mse'``, or a callable.

    Returns
    -------
    scorer : callable
        The resolved scoring function.
    """
    if callable(scoring):
        return scoring

    from tuiml.evaluation.metrics import accuracy_score, mean_squared_error, r2_score

    scorers = {
        "accuracy": accuracy_score,
        "r2": r2_score,
        "neg_mse": lambda a, b: -mean_squared_error(a, b),
    }
    if scoring not in scorers:
        raise ValueError(
            f"scoring must be one of {sorted(scorers)} or a callable, got {scoring!r}"
        )
    return scorers[scoring]


def permutation_importance(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    scoring: Union[str, Callable] = "accuracy",
    n_repeats: int = 10,
    feature_names: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> Explanation:
    """Measure importance by **shuffling a feature and watching the score fall**.

    A feature the model relies on cannot be scrambled without cost. Shuffling
    it breaks its relationship with the target while leaving its marginal
    distribution intact, so the drop in score is attributable to that feature
    and nothing else. Because it only needs predictions, it works on **any**
    model — and unlike a tree's built-in importance it measures what the model
    does on *held-out* data rather than how often a split happened to be used.

    Parameters
    ----------
    estimator : Algorithm
        A fitted model exposing ``predict``.
    X : np.ndarray of shape (n_samples, n_features)
        Data to evaluate on. Use a **held-out** set: on training data this
        measures what the model memorised, not what it generalises with.
    y : np.ndarray of shape (n_samples,)
        True targets for ``X``.
    scoring : str or callable, default='accuracy'
        Metric to degrade, higher being better. ``'accuracy'``, ``'r2'``,
        ``'neg_mse'``, or a ``(y_true, y_pred) -> float`` callable.
    n_repeats : int, default=10
        Shuffles per feature. The mean is the estimate and the spread says
        whether it is distinguishable from zero.
    feature_names : list of str, optional
        Names for the report.
    random_state : int, optional
        Seed for the shuffles.

    Returns
    -------
    explanation : Explanation
        ``values`` holds the mean score drop per feature. ``metadata`` carries
        ``std``, the per-repeat ``raw`` matrix and the unpermuted
        ``baseline_score``.

    Notes
    -----
    **Complexity.** ``n_features * n_repeats`` prediction passes; the model is
    never refitted.

    **Correlated features hide each other.** If two columns carry the same
    information, permuting either leaves the model able to recover the signal
    from the other, and *both* score as unimportant. The result is not that
    neither matters — it is that neither matters *given the other*. Group
    correlated columns and permute them together when that distinction
    matters, or use :func:`drop_column_importance`, which retrains and so
    answers a different question.

    Permuting also fabricates combinations the data never contains — a row
    with one feature's value drawn from a different row can be physically
    impossible — so the model is scored off its training distribution.

    References
    ----------
    .. [Breiman2001] Breiman, L. (2001). Random Forests. *Machine Learning*,
       45(1), 5-32. :doi:`10.1023/A:1010933404324`
    .. [Fisher2019] Fisher, A., Rudin, C., & Dominici, F. (2019). All Models
       are Wrong, but Many are Useful. *Journal of Machine Learning Research*,
       20(177), 1-81. :arxiv:`1801.01489`

    See Also
    --------
    :func:`~tuiml.explain.drop_column_importance` : Retrains without the feature; slower, answers "would I lose anything by not collecting this?"
    :class:`~tuiml.explain.TreeExplainer` : Exact per-sample attributions for tree models.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.explain import permutation_importance
    >>> from tuiml.algorithms.trees import RandomForestClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 4))
    >>> y = (X[:, 1] > 0).astype(int)          # only feature 1 matters
    >>> model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    >>> result = permutation_importance(X=X, y=y, estimator=model, random_state=0)
    >>> result.top(1)[0][0]
    'feature_1'
    """
    X = np.asarray(X)
    y = np.asarray(y)
    scorer = _resolve_scorer(scoring)
    rng = np.random.default_rng(random_state)

    baseline = float(scorer(y, estimator.predict(X)))
    raw = np.empty((X.shape[1], n_repeats))

    for column in range(X.shape[1]):
        for repeat in range(n_repeats):
            shuffled = X.copy()
            shuffled[:, column] = rng.permutation(shuffled[:, column])
            raw[column, repeat] = baseline - float(
                scorer(y, estimator.predict(shuffled))
            )

    return Explanation(
        values=raw.mean(axis=1),
        feature_names=feature_names,
        method="permutation_importance",
        metadata={
            "std": raw.std(axis=1),
            "raw": raw,
            "baseline_score": baseline,
            "scoring": scoring if isinstance(scoring, str) else "callable",
        },
    )


def drop_column_importance(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    scoring: Union[str, Callable] = "accuracy",
    cv: int = 3,
    feature_names: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> Explanation:
    """Measure importance by **removing a feature and refitting**.

    Permutation importance asks what a *fitted* model loses when a feature is
    corrupted. This asks the different and often more useful question: what
    would I lose by never collecting this feature at all? Because the model is
    refitted without it, it can compensate using whatever correlated columns
    remain — so a redundant feature correctly scores near zero, where
    permutation importance would also score it near zero but for the opposite
    reason.

    Parameters
    ----------
    estimator : Algorithm
        An **unfitted** model template; it is deep-copied and refitted once
        per feature.
    X : np.ndarray of shape (n_samples, n_features)
        Training data.
    y : np.ndarray of shape (n_samples,)
        Targets.
    scoring : str or callable, default='accuracy'
        Metric to degrade, higher being better.
    cv : int, default=3
        Folds used for each evaluation.
    feature_names : list of str, optional
        Names for the report.
    random_state : int, optional
        Seed for the fold split.

    Returns
    -------
    explanation : Explanation
        ``values`` holds the cross-validated score drop caused by dropping
        each feature. ``metadata`` carries ``baseline_score``.

    Notes
    -----
    **Complexity.** ``(n_features + 1) * cv`` model fits — far more expensive
    than permutation importance, which refits nothing. Reach for it when the
    decision at hand really is whether to collect a feature.

    A **negative** value is meaningful, not noise to clip: it says the model
    scored *better* without that feature, which happens when a column is pure
    noise the learner was overfitting to.

    References
    ----------
    .. [Fisher2019] Fisher, A., Rudin, C., & Dominici, F. (2019). All Models
       are Wrong, but Many are Useful. *Journal of Machine Learning Research*,
       20(177), 1-81. :arxiv:`1801.01489`

    See Also
    --------
    :func:`~tuiml.explain.permutation_importance` : No refitting; answers what the fitted model relies on.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.explain import drop_column_importance
    >>> from tuiml.algorithms.trees import DecisionTreeClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> result = drop_column_importance(
    ...     DecisionTreeClassifier(max_depth=4), X, y, cv=3, random_state=0)
    >>> result.top(1)[0][0]
    'feature_0'
    """
    from copy import deepcopy

    from tuiml.evaluation.splitting import KFold, StratifiedKFold

    X = np.asarray(X)
    y = np.asarray(y)
    scorer = _resolve_scorer(scoring)

    is_classification = y.dtype.kind in "iuOSUb" or len(np.unique(y)) < 20
    splitter = (
        StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        if is_classification
        else KFold(n_splits=cv, shuffle=True, random_state=random_state)
    )
    folds = list(splitter.split(X, y))

    def cross_validated(columns: np.ndarray) -> float:
        """Score the estimator restricted to the given columns."""
        scores = []
        for train_index, test_index in folds:
            model = deepcopy(estimator)
            model.fit(X[np.ix_(train_index, columns)], y[train_index])
            scores.append(
                float(scorer(y[test_index], model.predict(X[np.ix_(test_index, columns)])))
            )
        return float(np.mean(scores))

    all_columns = np.arange(X.shape[1])
    baseline = cross_validated(all_columns)

    drops = np.empty(X.shape[1])
    for column in range(X.shape[1]):
        remaining = np.delete(all_columns, column)
        # Dropping the only feature leaves nothing to fit on; the whole
        # baseline is attributable to it.
        drops[column] = (
            baseline - cross_validated(remaining) if len(remaining) else baseline
        )

    return Explanation(
        values=drops,
        feature_names=feature_names,
        method="drop_column_importance",
        metadata={"baseline_score": baseline},
    )
