"""Reduced Error Pruning Tree classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from tuiml.algorithms.trees._core import (
    TreeNode,
    TreeConfig,
    build_classifier_tree,
    build_regressor_tree,
    reduced_error_prune_classifier,
    reduced_error_prune_regressor,
    predict_single_numpy,
    predict_proba_single_numpy,
)

# Backward compat aliases
ReducedErrorPruningTreeClassifierNode = TreeNode
ReducedErrorPruningTreeRegressorNode = TreeNode

@classifier(tags=["trees", "pruning", "fast"], version="1.0.0")
class ReducedErrorPruningTreeClassifier(Classifier):
    """Reduced Error Pruning Tree classifier.

    ReducedErrorPruningTreeClassifier is a **fast** decision tree learner that
    builds a tree using **information gain** and prunes it using
    **reduced-error pruning** on a held-out validation set. This produces
    compact, well-generalized trees efficiently.

    Overview
    --------
    The algorithm proceeds in two phases:

    1. **Tree growing:** Build a full decision tree using information gain
       as the splitting criterion, evaluating all features and thresholds
    2. **Hold-out split:** Reserve a fraction of the data (determined by
       ``num_folds``) as a pruning validation set
    3. **Bottom-up pruning:** Starting from the leaves, replace each subtree
       with a leaf node if doing so does **not increase** the error on the
       validation set
    4. The result is a compact tree that avoids overfitting

    Theory
    ------
    The information gain for splitting set :math:`S` on attribute :math:`A`
    at threshold :math:`t` is:

    .. math::
        IG(S, A, t) = H(S) - \\frac{|S_L|}{|S|} H(S_L) - \\frac{|S_R|}{|S|} H(S_R)

    where :math:`H(S) = -\\sum_{i=1}^{c} p_i \\log_2(p_i)` is the entropy.

    During **reduced-error pruning**, a subtree rooted at node :math:`v` is
    replaced by a leaf if:

    .. math::
        \\text{err}_{leaf}(v) \\leq \\text{err}_{subtree}(v)

    where both errors are measured on the held-out validation set.

    Parameters
    ----------
    max_depth : int, default=-1
        Maximum depth of the tree (-1 for unlimited).
    min_samples_leaf : int, default=2
        Minimum samples required at a leaf node.
    min_variance_prop : float, default=1e-3
        Minimum proportion of variance for a split.
    num_folds : int, default=3
        Number of folds for pruning validation.
    no_pruning : bool, default=False
        If True, skip pruning.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    tree_ : TreeNode
        Root node of the tree.
    classes_ : np.ndarray
        Unique class labels.
    n_features_ : int
        Number of features seen during fit.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot m \\cdot \\log(n))` where :math:`n` = samples,
      :math:`m` = features
    - Pruning: :math:`O(n_{val} \\cdot d)` where :math:`n_{val}` = validation
      samples, :math:`d` = tree depth
    - Prediction: :math:`O(\\log(n))` per sample

    **When to use ReducedErrorPruningTreeClassifier:**

    - When you need a **fast** tree induction with built-in pruning
    - When a simple, well-pruned tree is preferred over complex models
    - As a base learner in ensemble methods that benefit from pruned trees
    - When training data is large enough to afford a validation split

    References
    ----------
    .. [Quinlan1987] Quinlan, J.R. (1987).
           **Simplifying Decision Trees.**
           *International Journal of Man-Machine Studies*, 27, pp. 221-234.
           DOI: `10.1016/S0020-7373(87)80053-6 <https://doi.org/10.1016/S0020-7373(87)80053-6>`_

    .. [Quinlan1993] Quinlan, J.R. (1993).
           **C4.5: Programs for Machine Learning.**
           *Morgan Kaufmann Publishers*.

    See Also
    --------
    :class:`~tuiml.algorithms.trees.C45TreeClassifier` : C4.5 tree with pessimistic pruning.
    :class:`~tuiml.algorithms.trees.RandomTreeClassifier` : Randomized tree without pruning.
    :class:`~tuiml.algorithms.trees.HoeffdingTreeClassifier` : Incremental tree for streaming data.

    Examples
    --------
    Basic usage with reduced-error pruning:

    >>> from tuiml.algorithms.trees import ReducedErrorPruningTreeClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]])
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>>
    >>> # Fit a REP tree
    >>> clf = ReducedErrorPruningTreeClassifier(max_depth=10)
    >>> clf.fit(X, y)
    ReducedErrorPruningTreeClassifier(...)
    >>> predictions = clf.predict(X)
    """

    def __init__(self, max_depth: int = -1,
                 min_samples_leaf: int = 2,
                 min_variance_prop: float = 1e-3,
                 num_folds: int = 3,
                 no_pruning: bool = False,
                 random_state: Optional[int] = None):
        """Initialize ReducedErrorPruningTreeClassifier classifier.

        Parameters
        ----------
        max_depth : int, default=-1
            Maximum tree depth (-1 for unlimited).
        min_samples_leaf : int, default=2
            Minimum samples at leaf.
        min_variance_prop : float, default=1e-3
            Minimum variance proportion for split.
        num_folds : int, default=3
            Folds for pruning validation.
        no_pruning : bool, default=False
            Skip pruning if True.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_variance_prop = min_variance_prop
        self.num_folds = num_folds
        self.no_pruning = no_pruning
        self.random_state = random_state
        self.tree_ = None
        self.classes_ = None
        self.n_features_ = None
        self._rng = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "max_depth": {
                "type": "integer",
                "default": -1,
                "description": "Maximum depth of the tree (-1 for unlimited)"
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "Minimum samples at leaf node"
            },
            "min_variance_prop": {
                "type": "number",
                "default": 1e-3,
                "minimum": 0,
                "description": "Minimum variance proportion for split"
            },
            "num_folds": {
                "type": "integer",
                "default": 3,
                "minimum": 2,
                "description": "Folds for pruning validation"
            },
            "no_pruning": {
                "type": "boolean",
                "default": False,
                "description": "Skip pruning if True"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return classifier capabilities."""
        return ["numeric", "missing_values", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m * log(n)) training, O(log(n)) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Quinlan, J.R. (1987). Simplifying Decision Trees. "
            "International Journal of Man-Machine Studies, 27, 221-234."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ReducedErrorPruningTreeClassifier":
        """Fit the ReducedErrorPruningTreeClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : ReducedErrorPruningTreeClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self._rng = np.random.RandomState(self.random_state)

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y])

        # Convert max_depth: -1 means unlimited → None for _core
        core_max_depth = None if self.max_depth <= 0 else self.max_depth

        config = TreeConfig(
            max_depth=core_max_depth,
            min_samples_split=2 * self.min_samples_leaf,
            min_samples_leaf=self.min_samples_leaf,
            criterion="entropy",
            n_classes=len(self.classes_),
        )

        if not self.no_pruning and len(y) > 10:
            n = len(y)
            val_size = max(1, n // self.num_folds)
            indices = self._rng.permutation(n)
            train_idx = indices[val_size:]
            val_idx = indices[:val_size]
            X_train, y_train = X[train_idx], y_idx[train_idx]
            X_val, y_val = X[val_idx], y_idx[val_idx]
        else:
            X_train, y_train = X, y_idx
            X_val, y_val = None, None

        self.tree_ = build_classifier_tree(X_train, y_train, config, self._rng)

        if not self.no_pruning and X_val is not None:
            self.tree_ = reduced_error_prune_classifier(
                self.tree_, X_val, y_val, self.classes_
            )

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        indices = np.array([
            int(np.argmax(predict_single_numpy(self.tree_, X[i])))
            for i in range(len(X))
        ])
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_classes = len(self.classes_)
        proba = np.zeros((len(X), n_classes))
        for i in range(len(X)):
            proba[i] = predict_proba_single_numpy(self.tree_, X[i], n_classes)
        return proba

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return f"ReducedErrorPruningTreeClassifier(n_features={self.n_features_})"
        return f"ReducedErrorPruningTreeClassifier(max_depth={self.max_depth})"


@regressor(tags=["trees", "pruning", "fast"], version="1.0.0")
class ReducedErrorPruningTreeRegressor(Regressor):
    """Reduced Error Pruning Tree regressor.

    ReducedErrorPruningTreeRegressor is a **fast** regression tree that
    builds a tree using **variance reduction** (MSE reduction) and prunes
    it using **reduced-error pruning** on a held-out validation set. This
    produces compact, well-generalized regression trees efficiently.

    Overview
    --------
    The algorithm proceeds in two phases:

    1. **Tree growing:** Build a full regression tree using variance
       reduction as the splitting criterion
    2. **Hold-out split:** Reserve a fraction of data (determined by
       ``num_folds``) as a pruning validation set
    3. **Bottom-up pruning:** Starting from the leaves, replace each
       subtree with a leaf if doing so does **not increase** the MSE
       on the validation set
    4. The result is a compact tree that avoids overfitting

    Theory
    ------
    The variance reduction for splitting set :math:`S` on attribute
    :math:`A` at threshold :math:`t` is:

    .. math::
        \\Delta \\text{MSE} = \\text{MSE}(S)
            - \\frac{|S_L|}{|S|} \\text{MSE}(S_L)
            - \\frac{|S_R|}{|S|} \\text{MSE}(S_R)

    During **reduced-error pruning**, a subtree rooted at node :math:`v`
    is replaced by a leaf if:

    .. math::
        \\text{MSE}_{leaf}(v) \\leq \\text{MSE}_{subtree}(v)

    where both MSE values are measured on the held-out validation set.

    Parameters
    ----------
    max_depth : int, default=-1
        Maximum depth of the tree (-1 for unlimited).
    min_samples_leaf : int, default=2
        Minimum samples required at a leaf node.
    min_variance_prop : float, default=1e-3
        Minimum proportion of variance for a split.
    num_folds : int, default=3
        Number of folds for pruning validation.
    no_pruning : bool, default=False
        If True, skip pruning.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    tree_ : TreeNode
        Root node of the tree.
    n_features_ : int
        Number of features seen during fit.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot m \\cdot \\log(n))` where :math:`n` = samples,
      :math:`m` = features
    - Pruning: :math:`O(n_{val} \\cdot d)` where :math:`n_{val}` = validation
      samples, :math:`d` = tree depth
    - Prediction: :math:`O(\\log(n))` per sample

    **When to use ReducedErrorPruningTreeRegressor:**

    - When you need a **fast** regression tree with built-in pruning
    - When a simple, well-pruned tree is preferred over complex models
    - As a base learner in ensemble methods that benefit from pruned trees

    References
    ----------
    .. [Quinlan1987] Quinlan, J.R. (1987).
           **Simplifying Decision Trees.**
           *International Journal of Man-Machine Studies*, 27, pp. 221-234.
           DOI: `10.1016/S0020-7373(87)80053-6 <https://doi.org/10.1016/S0020-7373(87)80053-6>`_

    See Also
    --------
    :class:`~tuiml.algorithms.trees.ReducedErrorPruningTreeClassifier` : Classifier counterpart.
    :class:`~tuiml.algorithms.trees.C45TreeRegressor` : Regression tree with pruning.
    :class:`~tuiml.algorithms.trees.RandomForestRegressor` : Ensemble of regression trees.

    Examples
    --------
    >>> from tuiml.algorithms.trees import ReducedErrorPruningTreeRegressor
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]])
    >>> y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5])
    >>> reg = ReducedErrorPruningTreeRegressor(max_depth=10)
    >>> reg.fit(X, y)
    ReducedErrorPruningTreeRegressor(...)
    >>> predictions = reg.predict(X)
    """

    def __init__(self, max_depth: int = -1,
                 min_samples_leaf: int = 2,
                 min_variance_prop: float = 1e-3,
                 num_folds: int = 3,
                 no_pruning: bool = False,
                 random_state: Optional[int] = None):
        """Initialize ReducedErrorPruningTreeRegressor.

        Parameters
        ----------
        max_depth : int, default=-1
            Maximum tree depth (-1 for unlimited).
        min_samples_leaf : int, default=2
            Minimum samples at leaf.
        min_variance_prop : float, default=1e-3
            Minimum variance proportion for split.
        num_folds : int, default=3
            Folds for pruning validation.
        no_pruning : bool, default=False
            Skip pruning if True.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_variance_prop = min_variance_prop
        self.num_folds = num_folds
        self.no_pruning = no_pruning
        self.random_state = random_state
        self.tree_ = None
        self.n_features_ = None
        self._rng = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "max_depth": {
                "type": "integer",
                "default": -1,
                "description": "Maximum depth of the tree (-1 for unlimited)"
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "Minimum samples at leaf node"
            },
            "min_variance_prop": {
                "type": "number",
                "default": 1e-3,
                "minimum": 0,
                "description": "Minimum variance proportion for split"
            },
            "num_folds": {
                "type": "integer",
                "default": 3,
                "minimum": 2,
                "description": "Folds for pruning validation"
            },
            "no_pruning": {
                "type": "boolean",
                "default": False,
                "description": "Skip pruning if True"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return regressor capabilities."""
        return ["numeric", "numeric_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m * log(n)) training, O(log(n)) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Quinlan, J.R. (1987). Simplifying Decision Trees. "
            "International Journal of Man-Machine Studies, 27, 221-234."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ReducedErrorPruningTreeRegressor":
        """Fit the ReducedErrorPruningTreeRegressor regression tree.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : ReducedErrorPruningTreeRegressor
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_ = X.shape[1]
        self._rng = np.random.RandomState(self.random_state)

        core_max_depth = None if self.max_depth <= 0 else self.max_depth

        config = TreeConfig(
            max_depth=core_max_depth,
            min_samples_split=2 * self.min_samples_leaf,
            min_samples_leaf=self.min_samples_leaf,
            criterion="squared_error",
        )

        if not self.no_pruning and len(y) > 10:
            n = len(y)
            val_size = max(1, n // self.num_folds)
            indices = self._rng.permutation(n)
            train_idx = indices[val_size:]
            val_idx = indices[:val_size]
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        self.tree_ = build_regressor_tree(X_train, y_train, config, self._rng)

        if not self.no_pruning and X_val is not None:
            self.tree_ = reduced_error_prune_regressor(
                self.tree_, X_val, y_val
            )

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.array([
            float(predict_single_numpy(self.tree_, X[i])[0])
            for i in range(len(X))
        ])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R-squared score.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.
        y : np.ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        r2 : float
            R-squared score.
        """
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=float)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1.0 - ss_res / ss_tot

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return f"ReducedErrorPruningTreeRegressor(n_features={self.n_features_})"
        return f"ReducedErrorPruningTreeRegressor(max_depth={self.max_depth})"
