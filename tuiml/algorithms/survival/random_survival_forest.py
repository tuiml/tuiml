"""Random survival forest: an ensemble of censoring-aware survival trees."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List

from tuiml.base.algorithms import Survival, survival
from tuiml.algorithms.trees import DecisionTreeRegressor


def _apply_flat(flat_tree, X):
    """Route samples to leaf node indices in a flattened tree.

    Parameters
    ----------
    flat_tree : FlattenedTree
        Flattened tree from a fitted ``DecisionTreeRegressor``.
    X : np.ndarray of shape (n_samples, n_features)
        Samples to route.

    Returns
    -------
    leaf : np.ndarray of shape (n_samples,)
        Leaf node index for each sample.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    leaf = np.empty(len(X), dtype=np.int64)
    for i in range(len(X)):
        node = 0
        while flat_tree.children_left[node] != -1:
            if X[i, flat_tree.feature[node]] <= flat_tree.threshold[node]:
                node = flat_tree.children_left[node]
            else:
                node = flat_tree.children_right[node]
        leaf[i] = node
    return leaf


def _nelson_aalen_at(time, event, horizon):
    """Return the Nelson-Aalen cumulative hazard of a sample up to ``horizon``.

    Parameters
    ----------
    time : np.ndarray of shape (n,)
        Observed times.
    event : np.ndarray of shape (n,)
        Event indicators.
    horizon : float
        Time point at which to evaluate the cumulative hazard.

    Returns
    -------
    H : float
        :math:`\\sum_{j: t_j \\leq \\text{horizon}} d_j / n_j`.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    event_times = np.unique(time[event == 1])
    H = 0.0
    for tj in event_times:
        if tj > horizon:
            break
        d = np.sum(event[time == tj])
        r = np.sum(time >= tj)
        H += d / r
    return float(H)


def _resolve_max_features(max_features, n_features):
    """Resolve the ``max_features`` specifier to an integer column count.

    Parameters
    ----------
    max_features : int, float, str or None
        Number of features to consider per tree. ``"sqrt"`` and ``"log2"``
        select the conventional defaults; an ``int`` is used directly; a
        ``float`` is a fraction; ``None`` uses all features.
    n_features : int
        Total number of features.

    Returns
    -------
    k : int
        Resolved feature count in ``[1, n_features]``.
    """
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError(f"Unknown max_features '{max_features}'")
    if isinstance(max_features, int):
        return int(np.clip(max_features, 1, n_features))
    if isinstance(max_features, float):
        return max(1, int(max_features * n_features))
    raise ValueError("max_features must be int, float, 'sqrt', 'log2' or None")


@survival(tags=["ensemble", "tree", "non_linear"], version="1.0.0")
class RandomSurvivalForest(Survival):
    """Random survival forest: an ensemble of survival trees.

    An ensemble of regression trees adapted to **right-censored** data. Each
    tree partitions the covariate space into homogeneous leaves; a leaf's risk
    is its **Nelson-Aalen cumulative hazard evaluated at a fixed horizon**, and
    the ensemble risk is the average leaf hazard across trees.

    Overview
    --------
    1. For each tree, bootstrap-sample the data and (optionally) subsample the
       features.
    2. Fit a :class:`~tuiml.algorithms.trees.DecisionTreeRegressor` to the
       observed times, which yields a variance-reduction partition of the
       covariate space (a cheap stand-in for the log-rank split of full RSF).
    3. Route the full training set through the tree and, per leaf, compute the
       Nelson-Aalen cumulative hazard of the leaf's ``(time, event)`` pairs,
       evaluated at a common horizon :math:`\\tau`.
    4. ``predict_risk`` routes a sample to one leaf per tree and averages the
       leaf hazards.

    Theory
    ------
    Within a leaf holding samples :math:`\\{(t_i, \\delta_i)\\}` the leaf risk is
    the Nelson-Aalen cumulative hazard up to a fixed horizon :math:`\\tau` (the
    median training event time):

    .. math::
        \\hat{H}_{\\text{leaf}}(\\tau) =
        \\sum_{j: t_j \\leq \\tau} \\frac{d_j}{n_j}.

    Evaluating at an intermediate :math:`\\tau` — rather than at :math:`\\infty` —
    is what makes the leaf risk sensitive to *when* events happen: a leaf whose
    members fail early has accumulated most of its hazard by :math:`\\tau`,
    whereas a leaf whose members fail late is still close to zero. For a sample
    :math:`x` landing in leaf :math:`\\ell_t(x)` of tree :math:`t`, the ensemble
    risk is

    .. math::
        \\text{risk}(x) = \\frac{1}{T} \\sum_{t=1}^{T}
        \\hat{H}_{\\ell_t(x)}(\\tau).

    A higher score means an earlier expected event, matching the ``Survival``
    convention.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of each tree (``None`` = unlimited).
    min_samples_split : int, default=2
        Minimum samples to split an internal node.
    min_samples_leaf : int, default=3
        Minimum samples required in a leaf (kept above 1 so every leaf has a
        stable hazard estimate).
    max_features : int, float, str or None, default="sqrt"
        Features to consider per tree. ``"sqrt"``, ``"log2"``, an ``int``, a
        ``float`` fraction, or ``None`` for all features.
    random_state : int or None, default=None
        Seed for reproducibility.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The fitted base trees.
    leaf_hazards_ : list of dict
        Mapping from leaf node index to cumulative hazard at ``horizon_``, one
        per tree.
    feature_subsets_ : list of np.ndarray
        Feature columns used by each tree.
    max_features_ : int
        Resolved number of features per tree.
    horizon_ : float
        Time horizon at which leaf hazards are evaluated (median training event
        time).
    n_features_in_ : int
        Number of features seen during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Fitting: :math:`O(T \\cdot n \\cdot p \\cdot n \\log n)` for the base
      trees plus :math:`O(T \\cdot n \\cdot k)` for leaf hazards.
    - Prediction: :math:`O(T \\cdot d)` per sample.

    **When to use RandomSurvivalForest:**

    - When the proportional-hazards assumption of Cox fails.
    - When covariate effects are non_linear or interactive.
    - Interpretability and coefficient inference are not required.

    References
    ----------
    .. [Ishwaran2008] Ishwaran, H., Kogalur, U.B., Blackstone, E.H. and Lauer,
           M.S. (2008). **Random Survival Forests.** *The Annals of Applied
           Statistics*, 2(3), 841-860.
           DOI: `10.1214/08-AOAS169 <https://doi.org/10.1214/08-AOAS169>`_
    .. [Breiman2001] Breiman, L. (2001). **Random Forests.** *Machine Learning*,
           45(1), 5-32.

    See Also
    --------
    :class:`~tuiml.algorithms.survival.CoxPHSurvival` : Semiparametric baseline.
    :class:`~tuiml.algorithms.survival.NelsonAalenEstimator` : Leaf hazard source.

    Examples
    --------
    >>> from tuiml.algorithms.survival import RandomSurvivalForest
    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> X = rng.normal(size=(40, 2))
    >>> time = np.exp(X[:, 0]) + rng.uniform(0, 1, size=40)
    >>> event = np.ones(40)
    >>> rsf = RandomSurvivalForest(n_estimators=10, random_state=0).fit(X, time, event)
    >>> rsf.predict_risk(X[:3]).shape
    (3,)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 3,
        max_features="sqrt",
        random_state: Optional[int] = None,
    ):
        """Initialize the random survival forest.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of trees.
        max_depth : int or None, default=None
            Maximum tree depth.
        min_samples_split : int, default=2
            Minimum samples to split a node.
        min_samples_leaf : int, default=3
            Minimum samples in a leaf.
        max_features : int, float, str or None, default="sqrt"
            Features per tree.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.estimators_ = None
        self.leaf_hazards_ = None
        self.feature_subsets_ = None
        self.max_features_ = None
        self.horizon_ = None
        self.n_features_in_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_estimators": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Number of trees in the forest",
            },
            "max_depth": {
                "type": ["integer", "null"],
                "default": None,
                "minimum": 1,
                "description": "Maximum depth of each tree (None = unlimited)",
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "minimum": 2,
                "description": "Minimum samples required to split a node",
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "description": "Minimum samples required at a leaf",
            },
            "max_features": {
                "oneOf": [
                    {"type": "integer", "minimum": 1},
                    {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    {"type": "string", "enum": ["sqrt", "log2"]},
                    {"type": "null"},
                ],
                "default": "sqrt",
                "description": "Number of features to consider per tree",
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility",
            },
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "censored", "non_linear", "ensemble", "tree"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return (
            "Fit: O(T * n * p * n*log(n)); "
            "predict: O(T * d) per sample, "
            "where T=n_estimators, d=tree depth"
        )

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Ishwaran, H., Kogalur, U.B., Blackstone, E.H. and Lauer, M.S., 2008. "
            "Random survival forests. Annals of Applied Statistics, 2(3), 841-860.",
            "Breiman, L., 2001. Random forests. Machine Learning, 45(1), 5-32."
        ]

    def fit(self, X, time, event) -> "RandomSurvivalForest":
        """Fit the forest on right-censored survival data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariate matrix.
        time : array-like of shape (n_samples,)
            Observed time (event or censoring).
        event : array-like of shape (n_samples,)
            Event indicator (1 = event observed, 0 = right-censored).

        Returns
        -------
        self : RandomSurvivalForest
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        time = np.asarray(time, dtype=np.float64).ravel()
        event = np.asarray(event, dtype=np.float64).ravel()
        n, p = X.shape

        if len(time) != n or len(event) != n:
            raise ValueError("X, time and event must describe the same number of samples")
        if np.any(time < 0):
            raise ValueError("time must be non-negative")
        if not np.all((event == 0) | (event == 1)):
            raise ValueError("event must contain only 0 and 1")

        self.n_features_in_ = p
        self.max_features_ = _resolve_max_features(self.max_features, p)
        event_mask = event == 1
        self.horizon_ = (
            float(np.median(time[event_mask])) if np.any(event_mask) else float(np.max(time))
        )
        rng = np.random.RandomState(self.random_state)

        self.estimators_ = []
        self.leaf_hazards_ = []
        self.feature_subsets_ = []

        for _ in range(self.n_estimators):
            cols = (
                np.sort(rng.choice(p, size=self.max_features_, replace=False))
                if self.max_features_ < p
                else np.arange(p)
            )
            self.feature_subsets_.append(cols)

            idx = rng.choice(n, size=n, replace=True)
            X_sub = X[idx][:, cols]

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(rng.randint(0, np.iinfo(np.int32).max)),
            ).fit(X_sub, time[idx])

            # Route the full training set so every leaf's hazard uses all
            # samples it would actually contain.
            leaf = _apply_flat(tree.flat_tree_, X[:, cols])
            leaf_hazard = {}
            for leaf_id in np.unique(leaf):
                mask = leaf == leaf_id
                leaf_hazard[int(leaf_id)] = _nelson_aalen_at(
                    time[mask], event[mask], self.horizon_
                )

            self.estimators_.append(tree)
            self.leaf_hazards_.append(leaf_hazard)

        self._is_fitted = True
        return self

    def predict_risk(self, X) -> np.ndarray:
        """Return the mean leaf cumulative hazard across trees.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariates.

        Returns
        -------
        risk : np.ndarray of shape (n_samples,)
            Ensemble risk. Higher values mean an earlier expected event.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        scores = np.zeros(len(X), dtype=np.float64)
        for tree, leaf_hazard, cols in zip(
            self.estimators_, self.leaf_hazards_, self.feature_subsets_
        ):
            leaf = _apply_flat(tree.flat_tree_, X[:, cols])
            scores += np.array(
                [leaf_hazard.get(int(li), 0.0) for li in leaf], dtype=np.float64
            )
        return scores / self.n_estimators
