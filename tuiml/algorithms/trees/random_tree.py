"""RandomTreeClassifier classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from tuiml.algorithms.trees._core import (
    TreeNode,
    TreeConfig,
    build_classifier_tree,
    build_regressor_tree,
    compute_max_features,
)
from tuiml.algorithms.trees._core_dispatch import (
    predict_batch_cpp as _predict_batch_cpp,
)

# Backward compat aliases
RandomTreeClassifierNode = TreeNode
RandomTreeRegressorNode = TreeNode

@classifier(tags=["trees", "random", "ensemble-component"], version="1.0.0")
class RandomTreeClassifier(Classifier):
    """Random Tree classifier - a single randomized decision tree.

    RandomTreeClassifier is a decision tree that considers a **random subset of
    attributes** at each node. This randomization, combined with growing
    the tree to full depth, makes it suitable as a **base learner** for
    Random Forest ensembles and other bagging methods.

    Overview
    --------
    The algorithm builds a fully-grown randomized tree:

    1. At each node, randomly select :math:`k` features from the full set
    2. Among the selected features, evaluate all candidate split points
       using **Gini impurity** reduction
    3. Choose the feature and threshold that maximize impurity decrease
    4. Recursively partition the data until stopping criteria are met
    5. No pruning is applied -- the tree is grown to full depth

    Theory
    ------
    The Gini impurity at a node with :math:`c` classes is:

    .. math::
        Gini(S) = 1 - \\sum_{i=1}^{c} p_i^2

    The impurity decrease for a split on attribute :math:`A` at threshold
    :math:`t` is:

    .. math::
        \\Delta Gini = Gini(S) - \\frac{|S_L|}{|S|} Gini(S_L) - \\frac{|S_R|}{|S|} Gini(S_R)

    The number of candidate features :math:`k` at each node is:

    - ``'sqrt'``: :math:`k = \\lfloor \\sqrt{m} \\rfloor`
    - ``'log2'``: :math:`k = \\lfloor \\log_2(m) \\rfloor`

    where :math:`m` is the total number of features.

    Parameters
    ----------
    max_features : {'sqrt', 'log2'}, int or float, default='sqrt'
        Number of features to consider at each split.
    max_depth : int, optional
        Maximum depth of the tree. None means unlimited.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    tree_ : TreeNode
        Root node of the decision tree.
    classes_ : np.ndarray
        Unique class labels.
    n_features_ : int
        Number of features seen during fit.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot k \\cdot \\log(n))` where :math:`n` = samples,
      :math:`k` = max_features
    - Prediction: :math:`O(\\log(n))` per sample (average tree depth)
    - Space: :math:`O(n)` for a fully-grown tree in the worst case

    **When to use RandomTreeClassifier:**

    - As a **base learner** inside Random Forest or bagging ensembles
    - When you need a single diverse tree for ensemble diversity
    - When fast training on high-dimensional data is required
    - Not recommended as a standalone classifier (high variance)

    References
    ----------
    .. [Breiman2001] Breiman, L. (2001).
           **Random Forests.**
           *Machine Learning*, 45(1), pp. 5-32.
           DOI: `10.1023/A:1010933404324 <https://doi.org/10.1023/A:1010933404324>`_

    See Also
    --------
    :class:`~tuiml.algorithms.trees.RandomForestClassifier` : Ensemble of random trees with bagging.
    :class:`~tuiml.algorithms.trees.C45TreeClassifier` : Deterministic C4.5 decision tree.
    :class:`~tuiml.algorithms.trees.DecisionStumpClassifier` : Single-split tree (weak learner).

    Examples
    --------
    Basic usage as a standalone classifier:

    >>> from tuiml.algorithms.trees import RandomTreeClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]])
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>>
    >>> # Fit a random tree
    >>> clf = RandomTreeClassifier(max_features='sqrt', random_state=42)
    >>> clf.fit(X, y)
    RandomTreeClassifier(...)
    >>> predictions = clf.predict(X)
    """

    def __init__(self, max_features: Any = 'sqrt',
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: Optional[int] = None):
        """Initialize RandomTreeClassifier classifier.

        Parameters
        ----------
        max_features : {'sqrt', 'log2'}, int or float, default='sqrt'
            Features to consider at each split. ``'sqrt'`` uses
            :math:`\\sqrt{m}`, ``'log2'`` uses :math:`\\log_2(m)`.
        max_depth : int or None, default=None
            Maximum tree depth.
        min_samples_split : int, default=2
            Minimum samples to split a node.
        min_samples_leaf : int, default=1
            Minimum samples at leaf node.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree_ = None
        self.classes_ = None
        self.n_features_ = None
        self._rng = None
        self._max_features_actual = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "max_features": {
                "type": ["string", "integer", "number"],
                "default": "sqrt",
                "description": "Number of features to consider at each split"
            },
            "max_depth": {
                "type": "integer",
                "default": None,
                "minimum": 1,
                "description": "Maximum depth of the tree"
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "minimum": 2,
                "description": "Minimum samples required to split a node"
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Minimum samples required at a leaf node"
            },
            "random_state": {
                "type": "integer",
                "default": None,
                "description": "Random seed for reproducibility"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return classifier capabilities."""
        return [
            "numeric",
            "missing_values",
            "binary_class",
            "multiclass"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m * log(n)) training, O(log(n)) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomTreeClassifier":
        """Fit the RandomTreeClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : RandomTreeClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self._rng = np.random.RandomState(self.random_state)
        self._max_features_actual = compute_max_features(self.max_features, self.n_features_)

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y])

        config = TreeConfig(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion="gini",
            max_features=self._max_features_actual,
            n_classes=len(self.classes_),
        )
        self.tree_ = build_classifier_tree(X, y_idx, config, self._rng)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

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

        raw = _predict_batch_cpp(self.tree_._cpp_flat, X)
        indices = np.argmax(raw, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

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

        return _predict_batch_cpp(self.tree_._cpp_flat, X)

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"RandomTreeClassifier(max_features={self._max_features_actual}, "
                   f"n_features={self.n_features_})")
        return f"RandomTreeClassifier(max_features={self.max_features})"


@regressor(tags=["trees", "random", "ensemble-component"], version="1.0.0")
class RandomTreeRegressor(Regressor):
    """Random Tree regressor - a single randomized regression tree.

    RandomTreeRegressor is a regression tree that considers a **random
    subset of features** at each node and uses **variance reduction**
    (MSE reduction) as the splitting criterion. Each leaf predicts the
    **mean target value** of its samples. Designed as a **base learner**
    for Random Forest regression ensembles.

    Overview
    --------
    The algorithm builds a fully-grown randomized regression tree:

    1. At each node, randomly select :math:`k` features from the full set
    2. Among the selected features, evaluate all candidate thresholds
       using **variance reduction** (MSE decrease)
    3. Choose the feature and threshold that maximize variance reduction
    4. Recursively partition until stopping criteria are met
    5. No pruning is applied -- the tree is grown to full depth

    Theory
    ------
    The variance (MSE) at a node with targets :math:`y_1, \\dots, y_n` is:

    .. math::
        \\text{MSE}(S) = \\frac{1}{|S|} \\sum_{i=1}^{|S|} (y_i - \\bar{y})^2

    The variance reduction for a split is:

    .. math::
        \\Delta \\text{MSE} = \\text{MSE}(S)
            - \\frac{|S_L|}{|S|} \\text{MSE}(S_L)
            - \\frac{|S_R|}{|S|} \\text{MSE}(S_R)

    Parameters
    ----------
    max_features : {'sqrt', 'log2'}, int or float, default='sqrt'
        Number of features to consider at each split.
    max_depth : int, optional
        Maximum depth of the tree. None means unlimited.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    tree_ : TreeNode
        Root node of the regression tree.
    n_features_ : int
        Number of features seen during fit.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot k \\cdot \\log(n))` where :math:`n` = samples,
      :math:`k` = max_features
    - Prediction: :math:`O(\\log(n))` per sample

    **When to use RandomTreeRegressor:**

    - As a **base learner** inside Random Forest regression ensembles
    - When you need a single diverse regression tree for ensemble diversity
    - Not recommended as a standalone regressor (high variance)

    References
    ----------
    .. [Breiman2001] Breiman, L. (2001).
           **Random Forests.**
           *Machine Learning*, 45(1), pp. 5-32.
           DOI: `10.1023/A:1010933404324 <https://doi.org/10.1023/A:1010933404324>`_

    See Also
    --------
    :class:`~tuiml.algorithms.trees.RandomForestRegressor` : Ensemble of random regression trees.
    :class:`~tuiml.algorithms.trees.RandomTreeClassifier` : Classifier counterpart.
    :class:`~tuiml.algorithms.trees.C45TreeRegressor` : Deterministic regression tree.

    Examples
    --------
    >>> from tuiml.algorithms.trees import RandomTreeRegressor
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]])
    >>> y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5])
    >>> reg = RandomTreeRegressor(max_features='sqrt', random_state=42)
    >>> reg.fit(X, y)
    RandomTreeRegressor(...)
    >>> predictions = reg.predict(X)
    """

    def __init__(self, max_features: Any = 'sqrt',
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: Optional[int] = None):
        """Initialize RandomTreeRegressor.

        Parameters
        ----------
        max_features : {'sqrt', 'log2'}, int or float, default='sqrt'
            Features to consider at each split.
        max_depth : int or None, default=None
            Maximum tree depth.
        min_samples_split : int, default=2
            Minimum samples to split a node.
        min_samples_leaf : int, default=1
            Minimum samples at leaf node.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree_ = None
        self.n_features_ = None
        self._rng = None
        self._max_features_actual = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "max_features": {
                "type": ["string", "integer", "number"],
                "default": "sqrt",
                "description": "Number of features to consider at each split"
            },
            "max_depth": {
                "type": "integer",
                "default": None,
                "minimum": 1,
                "description": "Maximum depth of the tree"
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "minimum": 2,
                "description": "Minimum samples required to split a node"
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Minimum samples required at a leaf node"
            },
            "random_state": {
                "type": "integer",
                "default": None,
                "description": "Random seed for reproducibility"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return regressor capabilities."""
        return ["numeric", "numeric_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * k * log(n)) training, O(log(n)) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomTreeRegressor":
        """Fit the RandomTreeRegressor regression tree.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RandomTreeRegressor
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_ = X.shape[1]
        self._rng = np.random.RandomState(self.random_state)
        self._max_features_actual = compute_max_features(self.max_features, self.n_features_)

        config = TreeConfig(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion="squared_error",
            max_features=self._max_features_actual,
        )
        self.tree_ = build_regressor_tree(X, y, config, self._rng)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples.

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

        raw = _predict_batch_cpp(self.tree_._cpp_flat, X)
        return raw[:, 0]

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
            return (f"RandomTreeRegressor(max_features={self._max_features_actual}, "
                   f"n_features={self.n_features_})")
        return f"RandomTreeRegressor(max_features={self.max_features})"
