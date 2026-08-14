"""RuleFit: a forest distilled into a sparse linear model of readable rules.

RuleFit fits a random forest, walks every root-to-leaf path to collect
conjunctions of the form ``feature <= threshold`` / ``feature > threshold``,
and then fits a sparse linear model (L1 / L2 penalised least squares via
coordinate descent) over **the rules plus the original features**. The result
is a small, human-readable set of rules with coefficients, keeping the
predictive power of the forest but with linear-model interpretability.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from tuiml.algorithms.trees import RandomForestClassifier, RandomForestRegressor


# ---------------------------------------------------------------------------
# Rule extraction and evaluation
# ---------------------------------------------------------------------------

def _as_2d(X) -> np.ndarray:
    """Return ``X`` as a 2-D float array, treating 1-D input as one column.

    Parameters
    ----------
    X : array-like
        Feature data, either 2-D or a single 1-D feature.

    Returns
    -------
    X2d : np.ndarray of shape (n_samples, n_features)
        Two-dimensional copy of the input.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return np.array(X, dtype=float)


def _extract_paths(node, path: List[Tuple[int, str, float]],
                   out: List[Tuple[Tuple[int, str, float], ...]]) -> None:
    """Recursively collect root-to-leaf conjunctions from a ``TreeNode``.

    Parameters
    ----------
    node : TreeNode or None
        Current node.
    path : list of tuple
        Conditions accumulated along the path so far.
    out : list of tuple
        Destination for completed rules.
    """
    if node is None:
        return
    if node.is_leaf:
        if path:
            out.append(tuple(path))
        return
    threshold = float(node.threshold)
    _extract_paths(node.left, path + [(int(node.feature_index), "<=", threshold)], out)
    _extract_paths(node.right, path + [(int(node.feature_index), ">", threshold)], out)


def _collect_rules(estimators, max_rules: Optional[int]) -> List[Tuple[Tuple[int, str, float], ...]]:
    """Extract and deduplicate rules from a forest, most frequent first.

    Parameters
    ----------
    estimators : list of TreeNode
        Fitted tree roots.
    max_rules : int or None
        Cap on the number of rules returned (``None`` = unlimited).

    Returns
    -------
    rules : list of tuple
        Rules as tuples of ``(feature_index, op, threshold)`` conditions.
    """
    counts: Counter = Counter()
    for tree in estimators:
        paths: List[Tuple[Tuple[int, str, float], ...]] = []
        _extract_paths(tree, [], paths)
        for path in paths:
            counts[path] += 1
    rules = [rule for rule, _ in counts.most_common()]
    if max_rules is not None:
        rules = rules[:max_rules]
    return rules


def _rule_indicator(rule: Tuple[Tuple[int, str, float], ...], X: np.ndarray) -> np.ndarray:
    """Evaluate a rule as a 0/1 indicator over all samples.

    Parameters
    ----------
    rule : tuple of tuple
        Conjunction of ``(feature_index, op, threshold)`` conditions.
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.

    Returns
    -------
    indicator : np.ndarray of shape (n_samples,)
        1.0 where every condition holds, else 0.0.
    """
    mask = np.ones(X.shape[0], dtype=bool)
    for feature_index, op, threshold in rule:
        col = X[:, feature_index]
        if op == "<=":
            mask &= col <= threshold
        else:
            mask &= col > threshold
    return mask.astype(np.float64)


def _rule_to_string(rule, feature_names: Optional[List[str]]) -> str:
    """Render a rule as a human-readable string.

    Parameters
    ----------
    rule : tuple of tuple
        Conjunction of conditions.
    feature_names : list of str or None
        Optional feature names.

    Returns
    -------
    text : str
        e.g. ``"feature_0 > 1.5 & feature_2 <= -0.25"``.
    """
    parts = []
    for feature_index, op, threshold in rule:
        name = feature_names[feature_index] if feature_names is not None else f"feature_{feature_index}"
        parts.append(f"{name} {op} {threshold:g}")
    return " & ".join(parts)


# ---------------------------------------------------------------------------
# Sparse linear solver (coordinate descent)
# ---------------------------------------------------------------------------

def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center and scale the columns of a design matrix.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Design matrix.

    Returns
    -------
    Xs : np.ndarray
        Standardised design matrix.
    mu : np.ndarray of shape (n_features,)
        Column means.
    sigma : np.ndarray of shape (n_features,)
        Column standard deviations (constant columns get 1.0).
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (X - mu) / sigma, mu, sigma


def _soft_threshold(z: np.ndarray, gamma: float) -> np.ndarray:
    """Soft-thresholding operator for L1 coordinate descent."""
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)


def _coordinate_descent(X: np.ndarray, y: np.ndarray, alpha: float,
                        l1_ratio: float, max_iter: int, tol: float) -> np.ndarray:
    """Solve a penalised least-squares problem by cyclic coordinate descent.

    Minimises :math:`0.5 \\|Xw - y\\|^2 + \\alpha \\cdot l1\\_ratio \\|w\\|_1
    + 0.5 \\alpha (1 - l1\\_ratio) \\|w\\|^2_2`.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Standardised design matrix.
    y : np.ndarray of shape (n_samples,)
        Centered target.
    alpha : float
        Regularisation strength.
    l1_ratio : float
        Mix of L1 (lasso) vs L2 (ridge) penalty in ``[0, 1]``.
    max_iter : int
        Maximum passes over all coordinates.
    tol : float
        Convergence tolerance on the maximum weight change.

    Returns
    -------
    w : np.ndarray of shape (n_features,)
        Fitted weights.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    col_norms = (X * X).sum(axis=0)
    residual = y.astype(float).copy()
    l1_penalty = alpha * l1_ratio
    ridge = alpha * (1.0 - l1_ratio)
    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(n_features):
            col_norm = col_norms[j]
            if col_norm <= 0.0:
                continue
            rho = float(X[:, j] @ residual) + col_norm * w[j]
            if l1_penalty > 0.0:
                w_new = _soft_threshold(rho, l1_penalty) / (col_norm + ridge)
            else:
                w_new = rho / (col_norm + ridge)
            delta = w_new - w[j]
            if delta != 0.0:
                residual -= X[:, j] * delta
                w[j] = w_new
                max_delta = max(max_delta, abs(delta))
        if max_delta < tol:
            break
    return w


class _RuleFitBase:
    """Shared forest distillation and sparse linear fit for RuleFit models.

    Internal mixin; not registered and not instantiated directly.
    """

    def _fit_forest(self, X: np.ndarray, y: np.ndarray):
        """Fit the underlying random forest and return its trees.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target (labels for classification, values for regression).

        Returns
        -------
        estimators : list of TreeNode
            Fitted tree roots.
        """
        if self._task == "classification":
            forest = RandomForestClassifier(
                n_estimators=self.n_estimators, max_depth=self.tree_size,
                random_state=self.random_state, n_jobs=1,
            )
        else:
            forest = RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.tree_size,
                random_state=self.random_state, n_jobs=1,
            )
        forest.fit(X, y)
        return forest.estimators_

    def _fit_linear(self, design: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit the penalised linear model on a standardised design matrix.

        Parameters
        ----------
        design : np.ndarray of shape (n_samples, n_rules + n_features)
            Raw design matrix (rule indicators then original features).
        y : np.ndarray of shape (n_samples,)
            Continuous target (0/1 encoded for classification).

        Returns
        -------
        coef : np.ndarray of shape (n_rules + n_features,)
            Coefficients on the *original* design scale.
        intercept : float
            Intercept on the original target scale.
        """
        y = np.asarray(y, dtype=float).ravel()
        Xs, mu, sigma = _standardize(design)
        y_mean = float(y.mean())
        y_centered = y - y_mean
        w_std = _coordinate_descent(
            Xs, y_centered, self.alpha, self._l1_ratio, self.max_iter, self.tol,
        )
        coef = w_std / sigma
        intercept = y_mean - float(np.sum(w_std * mu / sigma))
        return coef, intercept

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        """Build the rule + feature design matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        design : np.ndarray of shape (n_samples, n_rules + n_features)
            Rule indicators followed by the original features.
        """
        X = _as_2d(X)
        rule_cols = [_rule_indicator(rule, X) for rule in self._rules_]
        if rule_cols:
            rules_mat = np.column_stack(rule_cols)
        else:
            rules_mat = np.empty((X.shape[0], 0))
        return np.hstack([rules_mat, X])

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        """Compute the linear score (intercept + rules + features).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        score : np.ndarray of shape (n_samples,)
            Linear score before any link function.
        """
        X = _as_2d(X)
        score = np.full(X.shape[0], self.intercept_, dtype=float)
        for i, rule in enumerate(self._rules_):
            coef = self.rule_coefs_[i]
            if coef != 0.0:
                score += coef * _rule_indicator(rule, X)
        score += X @ self.feature_coefs_
        return score

    def get_rules(self, min_abs_coef: float = 1e-8) -> List[Tuple[str, float]]:
        """Return the learned rules and their coefficients, most important first.

        Parameters
        ----------
        min_abs_coef : float, default=1e-8
            Rules whose absolute coefficient is below this are dropped.

        Returns
        -------
        rules : list of tuple (str, float)
            ``(rule_string, coefficient)`` sorted by descending absolute
            coefficient.
        """
        self._check_is_fitted()
        pairs = [(self.rules_[i], float(self.rule_coefs_[i]))
                 for i in range(len(self.rules_))
                 if abs(self.rule_coefs_[i]) >= min_abs_coef]
        pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return pairs

    def _finalize(self, X: np.ndarray, y: np.ndarray) -> None:
        """Extract rules, fit the sparse linear model, store interpretable state.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Continuous target (0/1 encoded for classification).
        """
        estimators = self._fit_forest(X, y)
        self._rules_ = _collect_rules(estimators, self.max_rules)
        self.rules_ = [_rule_to_string(rule, self.feature_names) for rule in self._rules_]
        design = self._design_matrix(X)
        coef, intercept = self._fit_linear(design, y)
        n_rules = len(self._rules_)
        self.rule_coefs_ = coef[:n_rules]
        self.feature_coefs_ = coef[n_rules:]
        self.intercept_ = float(intercept)
        self.n_rules_ = n_rules
        self.n_features_ = X.shape[1]
        self._is_fitted = True


@regressor(tags=["glassbox", "interpretable", "rule-based", "ensemble", "linear"], version="1.0.0")
class RuleFitRegressor(Regressor, _RuleFitBase):
    """RuleFit regressor: forest rules distilled into a sparse linear model.

    A **glassbox** regressor that fits a random forest, extracts human-readable
    conjunctions from its root-to-leaf paths, and then fits a sparse linear
    model over those rules plus the original features.

    Overview
    --------
    1. Fit a :class:`~tuiml.algorithms.trees.RandomForestRegressor`
    2. Walk each tree, turning every root-to-leaf path into a conjunction of
       ``feature <= t`` / ``feature > t`` conditions (a *rule*)
    3. Deduplicate rules, optionally capping to the most frequent
    4. Build a design matrix of rule indicators plus the raw features
    5. Fit a penalised linear model (L1 / L2) over that design matrix via
       coordinate descent

    Theory
    ------
    The prediction is the linear form

    .. math::
        \\hat{y}(x) = \\beta_0 + \\sum_{r} a_r \\cdot r(x) + \\sum_{j} b_j x_j

    where :math:`r(x) \\in \\{0, 1\\}` is a rule indicator. The coefficients
    are found by minimising

    .. math::
        \\min_{a, b} \\frac{1}{2}\\|\\hat{y} - y\\|^2
        + \\alpha \\cdot l1\\_ratio \\|(a, b)\\|_1
        + \\frac{\\alpha (1 - l1\\_ratio)}{2} \\|(a, b)\\|^2_2

    so an L1 penalty drives most rule coefficients to exactly zero, leaving a
    small readable rule set.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the underlying forest.
    tree_size : int, default=3
        Maximum depth of each tree, which bounds rule length.
    max_rules : int or None, default=None
        Cap on the number of distinct rules kept (``None`` = unlimited).
    penalty : {'l1', 'l2'}, default='l1'
        Sparse (lasso) or ridge regularisation for the linear model.
    alpha : float, default=0.1
        Regularisation strength.
    max_iter : int, default=1000
        Maximum coordinate-descent passes.
    tol : float, default=1e-4
        Convergence tolerance on the maximum weight change.
    random_state : int or None, default=None
        Seed for the forest (reproducible fits).
    feature_names : list of str, optional
        Names used in the printed rules.

    Attributes
    ----------
    rules_ : list of str
        Human-readable rule strings (deduplicated).
    rule_coefs_ : np.ndarray of shape (n_rules,)
        Coefficient of each rule.
    feature_coefs_ : np.ndarray of shape (n_features,)
        Coefficient of each original feature.
    intercept_ : float
        Linear intercept.
    n_rules_ : int
        Number of distinct rules extracted.
    n_features_ : int
        Number of features seen during fit.

    Notes
    -----
    **Complexity:**

    - Training: forest fit :math:`O(T \\cdot n \\cdot \\log n)` plus
      coordinate descent :math:`O(\\text{max\\_iter} \\cdot n \\cdot p)` with
      :math:`p` = rules + features.
    - Prediction: :math:`O(p)` per sample.

    **When to use RuleFitRegressor:**

    - When you want a small set of readable ``if feature > t`` rules
    - When the target has threshold effects a plain linear model misses
    - When you need sparse, auditable coefficients

    References
    ----------
    .. [Friedman2008] Friedman, J.H. and Popescu, B.E. (2008).
           **Predictive learning via rule ensembles.**
           *The Annals of Applied Statistics*, 2(3), 916-954.
           DOI: `10.1214/07-AOAS148 <https://doi.org/10.1214/07-AOAS148>`_

    See Also
    --------
    :class:`~tuiml.algorithms.glassbox.RuleFitClassifier` : Classification counterpart.
    :class:`~tuiml.algorithms.trees.RandomForestRegressor` : The forest being distilled.

    Examples
    --------
    >>> from tuiml.algorithms.glassbox import RuleFitRegressor
    >>> import numpy as np
    >>> X = np.array([[i] for i in range(40)], dtype=float)
    >>> y = np.where(X.ravel() < 20.0, 0.0, 5.0)
    >>> reg = RuleFitRegressor(n_estimators=50, tree_size=2, random_state=0)
    >>> _ = reg.fit(X, y)
    >>> float(np.abs(reg.predict(np.array([[25.0]]))[0] - 5.0)) < 1.5
    True
    >>> isinstance(reg.get_rules(), list) and len(reg.get_rules()) > 0
    True
    """

    def __init__(self, n_estimators: int = 100, tree_size: int = 3,
                 max_rules: Optional[int] = None, penalty: str = "l1",
                 alpha: float = 0.1, max_iter: int = 1000, tol: float = 1e-4,
                 random_state: Optional[int] = None,
                 feature_names: Optional[List[str]] = None):
        """Initialize RuleFitRegressor.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of trees in the forest.
        tree_size : int, default=3
            Maximum tree depth.
        max_rules : int or None, default=None
            Cap on distinct rules kept.
        penalty : {'l1', 'l2'}, default='l1'
            Regularisation type.
        alpha : float, default=0.1
            Regularisation strength.
        max_iter : int, default=1000
            Maximum coordinate-descent passes.
        tol : float, default=1e-4
            Convergence tolerance.
        random_state : int or None, default=None
            Forest seed.
        feature_names : list of str, optional
            Names used in rule strings.
        """
        super().__init__()
        if penalty not in ("l1", "l2"):
            raise ValueError("penalty must be 'l1' or 'l2'")
        self.n_estimators = n_estimators
        self.tree_size = tree_size
        self.max_rules = max_rules
        self.penalty = penalty
        self._l1_ratio = 1.0 if penalty == "l1" else 0.0
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.feature_names = feature_names
        self._task = "regression"
        self._rules_ = None
        self.rules_ = None
        self.rule_coefs_ = None
        self.feature_coefs_ = None
        self.intercept_ = None
        self.n_rules_ = None
        self.n_features_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_estimators": {"type": "integer", "default": 100, "minimum": 1,
                             "description": "Number of trees in the forest"},
            "tree_size": {"type": "integer", "default": 3, "minimum": 1,
                          "description": "Maximum depth of each tree"},
            "max_rules": {"type": ["integer", "null"], "default": None, "minimum": 1,
                          "description": "Cap on distinct rules kept"},
            "penalty": {"type": "string", "default": "l1", "enum": ["l1", "l2"],
                        "description": "Regularisation type"},
            "alpha": {"type": "number", "default": 0.1, "minimum": 0.0,
                      "description": "Regularisation strength"},
            "max_iter": {"type": "integer", "default": 1000, "minimum": 1,
                         "description": "Maximum coordinate-descent passes"},
            "tol": {"type": "number", "default": 1e-4, "minimum": 0.0,
                    "description": "Convergence tolerance"},
            "random_state": {"type": ["integer", "null"], "default": None,
                             "description": "Forest random seed"},
            "feature_names": {"type": ["array", "null"], "default": None,
                              "description": "Optional feature names"},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return regressor capabilities."""
        return ["numeric", "regression"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n_estimators * n * log(n)) forest fit + O(max_iter * n * (rules + features)) coordinate descent"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Friedman, J.H. & Popescu, B.E. (2008). Predictive learning via "
            "rule ensembles. The Annals of Applied Statistics, 2(3), 916-954."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RuleFitRegressor":
        """Fit the rule ensemble.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RuleFitRegressor
            Fitted regressor.
        """
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).ravel()
        self._finalize(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        self._check_is_fitted()
        return self._raw_predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the R-squared score on the given data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.
        y : np.ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        r2 : float
            R-squared score.
        """
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot == 0.0:
            return 0.0
        return 1.0 - ss_res / ss_tot

    def __repr__(self) -> str:
        if self._is_fitted:
            return (f"RuleFitRegressor(n_estimators={self.n_estimators}, "
                    f"n_rules={self.n_rules_})")
        return f"RuleFitRegressor(n_estimators={self.n_estimators})"


@classifier(tags=["glassbox", "interpretable", "rule-based", "ensemble", "linear"], version="1.0.0")
class RuleFitClassifier(Classifier, _RuleFitBase):
    """RuleFit classifier: forest rules distilled into a sparse linear model.

    A **glassbox** binary classifier that fits a random forest, extracts
    human-readable conjunctions from its root-to-leaf paths, and fits a sparse
    linear model over those rules plus the original features. The linear score
    is interpreted as a class-1 probability (linear probability model).

    Overview
    --------
    1. Fit a :class:`~tuiml.algorithms.trees.RandomForestClassifier`
    2. Walk each tree, turning every root-to-leaf path into a conjunction of
       ``feature <= t`` / ``feature > t`` conditions (a *rule*)
    3. Deduplicate rules, optionally capping to the most frequent
    4. Encode the binary labels as 0/1 and build a design matrix of rule
       indicators plus the raw features
    5. Fit a penalised linear model (L1 / L2) over that design matrix via
       coordinate descent and interpret the score as a class-1 probability

    Theory
    ------
    The score is the linear form

    .. math::
        s(x) = \\beta_0 + \\sum_{r} a_r \\cdot r(x) + \\sum_{j} b_j x_j

    interpreted as the class-1 probability (a linear probability model, the
    approach of the original RuleFit paper), clipped to :math:`[0, 1]`.
    Coefficients minimise a penalised least-squares objective on the 0/1
    target, with L1 shrinkage yielding a sparse, readable rule set.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the underlying forest.
    tree_size : int, default=3
        Maximum depth of each tree, which bounds rule length.
    max_rules : int or None, default=None
        Cap on the number of distinct rules kept (``None`` = unlimited).
    penalty : {'l1', 'l2'}, default='l1'
        Sparse (lasso) or ridge regularisation for the linear model.
    alpha : float, default=0.1
        Regularisation strength.
    max_iter : int, default=1000
        Maximum coordinate-descent passes.
    tol : float, default=1e-4
        Convergence tolerance on the maximum weight change.
    random_state : int or None, default=None
        Seed for the forest (reproducible fits).
    feature_names : list of str, optional
        Names used in the printed rules.

    Attributes
    ----------
    rules_ : list of str
        Human-readable rule strings (deduplicated).
    rule_coefs_ : np.ndarray of shape (n_rules,)
        Coefficient of each rule.
    feature_coefs_ : np.ndarray of shape (n_features,)
        Coefficient of each original feature.
    intercept_ : float
        Linear intercept.
    classes_ : np.ndarray
        The two class labels in sorted order.
    n_rules_ : int
        Number of distinct rules extracted.
    n_features_ : int
        Number of features seen during fit.

    Notes
    -----
    **Complexity:**

    - Training: forest fit :math:`O(T \\cdot n \\cdot \\log n)` plus
      coordinate descent :math:`O(\\text{max\\_iter} \\cdot n \\cdot p)` with
      :math:`p` = rules + features.
    - Prediction: :math:`O(p)` per sample.

    **When to use RuleFitClassifier:**

    - Binary classification where a small readable rule set is required
    - When the decision boundary has threshold effects a logistic model misses
    - When you want sparse, auditable coefficients

    References
    ----------
    .. [Friedman2008] Friedman, J.H. and Popescu, B.E. (2008).
           **Predictive learning via rule ensembles.**
           *The Annals of Applied Statistics*, 2(3), 916-954.
           DOI: `10.1214/07-AOAS148 <https://doi.org/10.1214/07-AOAS148>`_

    See Also
    --------
    :class:`~tuiml.algorithms.glassbox.RuleFitRegressor` : Regression counterpart.
    :class:`~tuiml.algorithms.trees.RandomForestClassifier` : The forest being distilled.

    Examples
    --------
    >>> from tuiml.algorithms.glassbox import RuleFitClassifier
    >>> import numpy as np
    >>> X = np.array([[i] for i in range(40)], dtype=float)
    >>> y = np.where(X.ravel() < 20.0, 0, 1)
    >>> clf = RuleFitClassifier(n_estimators=50, tree_size=2, random_state=0)
    >>> _ = clf.fit(X, y)
    >>> clf.predict(np.array([[5.0], [35.0]])).tolist()
    [0, 1]
    >>> isinstance(clf.get_rules(), list) and len(clf.get_rules()) > 0
    True
    """

    def __init__(self, n_estimators: int = 100, tree_size: int = 3,
                 max_rules: Optional[int] = None, penalty: str = "l1",
                 alpha: float = 0.1, max_iter: int = 1000, tol: float = 1e-4,
                 random_state: Optional[int] = None,
                 feature_names: Optional[List[str]] = None):
        """Initialize RuleFitClassifier.

        Parameters
        ----------
        n_estimators : int, default=100
            Number of trees in the forest.
        tree_size : int, default=3
            Maximum tree depth.
        max_rules : int or None, default=None
            Cap on distinct rules kept.
        penalty : {'l1', 'l2'}, default='l1'
            Regularisation type.
        alpha : float, default=0.1
            Regularisation strength.
        max_iter : int, default=1000
            Maximum coordinate-descent passes.
        tol : float, default=1e-4
            Convergence tolerance.
        random_state : int or None, default=None
            Forest seed.
        feature_names : list of str, optional
            Names used in rule strings.
        """
        super().__init__()
        if penalty not in ("l1", "l2"):
            raise ValueError("penalty must be 'l1' or 'l2'")
        self.n_estimators = n_estimators
        self.tree_size = tree_size
        self.max_rules = max_rules
        self.penalty = penalty
        self._l1_ratio = 1.0 if penalty == "l1" else 0.0
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.feature_names = feature_names
        self._task = "classification"
        self._rules_ = None
        self.rules_ = None
        self.rule_coefs_ = None
        self.feature_coefs_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_rules_ = None
        self.n_features_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_estimators": {"type": "integer", "default": 100, "minimum": 1,
                             "description": "Number of trees in the forest"},
            "tree_size": {"type": "integer", "default": 3, "minimum": 1,
                          "description": "Maximum depth of each tree"},
            "max_rules": {"type": ["integer", "null"], "default": None, "minimum": 1,
                          "description": "Cap on distinct rules kept"},
            "penalty": {"type": "string", "default": "l1", "enum": ["l1", "l2"],
                        "description": "Regularisation type"},
            "alpha": {"type": "number", "default": 0.1, "minimum": 0.0,
                      "description": "Regularisation strength"},
            "max_iter": {"type": "integer", "default": 1000, "minimum": 1,
                         "description": "Maximum coordinate-descent passes"},
            "tol": {"type": "number", "default": 1e-4, "minimum": 0.0,
                    "description": "Convergence tolerance"},
            "random_state": {"type": ["integer", "null"], "default": None,
                             "description": "Forest random seed"},
            "feature_names": {"type": ["array", "null"], "default": None,
                              "description": "Optional feature names"},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return classifier capabilities."""
        return ["numeric", "binary_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n_estimators * n * log(n)) forest fit + O(max_iter * n * (rules + features)) coordinate descent"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Friedman, J.H. & Popescu, B.E. (2008). Predictive learning via "
            "rule ensembles. The Annals of Applied Statistics, 2(3), 916-954."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RuleFitClassifier":
        """Fit the rule ensemble.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels (must be exactly two classes).

        Returns
        -------
        self : RuleFitClassifier
            Fitted classifier.
        """
        X = _as_2d(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(
                "RuleFitClassifier supports binary classification only; "
                f"got {self.classes_.size} classes."
            )
        y01 = np.where(y == self.classes_[1], 1.0, 0.0)
        self._finalize(X, y01)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Uses the linear-probability interpretation of RuleFit: the raw score
        is treated as the class-1 probability and clipped to ``[0, 1]``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, 2)
            Probabilities for classes 0 and 1.
        """
        self._check_is_fitted()
        score = self._raw_predict(X)
        p = np.clip(score, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def __repr__(self) -> str:
        if self._is_fitted:
            return (f"RuleFitClassifier(n_estimators={self.n_estimators}, "
                    f"n_rules={self.n_rules_})")
        return f"RuleFitClassifier(n_estimators={self.n_estimators})"
