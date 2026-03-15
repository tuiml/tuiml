"""CART Decision Tree for classification and regression."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor
from tuiml.algorithms.trees._core import (
    TreeNode,
    flatten_tree,
    count_nodes,
    max_depth_of,
    get_tree_description,
    predict_batch,
    predict_proba_batch,
    TreeConfig,
    build_classifier_tree,
    build_regressor_tree,
    cost_complexity_prune,
)

# Backward compat aliases
CARTNode = TreeNode
FlattenedTree = __import__(
    "tuiml.algorithms.trees._core.nodes", fromlist=["FlattenedTree"]
).FlattenedTree


# =========================================================================
# DecisionTreeClassifier
# =========================================================================

@classifier(tags=["trees", "interpretable"], version="1.0.0")
class DecisionTreeClassifier(Classifier):
    """CART decision tree for classification.

    A **CART** (Classification and Regression Trees) classifier that uses
    vectorised batch prediction.

    Overview
    --------
    1. Build the tree recursively using the selected **impurity criterion**
       (Gini, entropy, or log-loss) to evaluate candidate splits
    2. After fitting, **flatten** the tree into parallel arrays
    3. Predict via efficient tree traversal

    Theory
    ------
    **Gini impurity** (``criterion="gini"``):

    .. math::
        G(S) = 1 - \\sum_{k=1}^{C} p_k^2

    **Entropy / information gain** (``criterion="entropy"`` or ``"log_loss"``):

    .. math::
        H(S) = -\\sum_{k=1}^{C} p_k \\log_2 p_k

    **Gain ratio** (``criterion="gain_ratio"``, C4.5):

    .. math::
        \\text{GainRatio} = \\frac{\\Delta H}{\\text{SplitInfo}}, \\quad
        \\text{SplitInfo} = -\\sum_i \\frac{|S_i|}{|S|} \\log_2 \\frac{|S_i|}{|S|}

    where :math:`p_k` is the proportion of class :math:`k` in the node.
    A split is chosen to maximise the **weighted impurity reduction**
    (or gain ratio for C4.5):

    .. math::
        \\Delta I = I(S) - \\frac{|S_L|}{|S|} I(S_L)
        - \\frac{|S_R|}{|S|} I(S_R)

    Optional **minimal cost-complexity pruning** removes branches whose
    effective :math:`\\alpha` is below ``ccp_alpha``.

    Parameters
    ----------
    criterion : str, default="gini"
        The function to measure the quality of a split. Supported criteria:
        ``"gini"`` for the Gini impurity (CART), ``"entropy"`` for the
        information gain (ID3), ``"log_loss"`` (alias for entropy), and
        ``"gain_ratio"`` for the C4.5 gain-ratio criterion.
    max_depth : int or None, default=None
        Maximum depth of the tree. ``None`` means unlimited.
    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node.
    min_impurity_decrease : float, default=0.0
        A node is split only if the impurity decrease is at least this value.
    ccp_alpha : float, default=0.0
        Complexity parameter for minimal cost-complexity pruning. Subtrees
        with effective alpha less than ``ccp_alpha`` are pruned.
    random_state : int or None, default=None
        Random seed for reproducibility when features are tied.

    Attributes
    ----------
    tree_ : TreeNode
        The root node of the recursive tree (training representation).
    flat_tree_ : FlattenedTree
        Flattened parallel-array tree used for JIT prediction.
    classes_ : np.ndarray
        Unique class labels discovered during ``fit()``.
    n_classes_ : int
        Number of classes.
    n_features_ : int
        Number of features seen during ``fit()``.
    max_depth_ : int
        Actual depth of the fitted tree.
    n_nodes_ : int
        Total number of nodes in the fitted tree.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot m \\cdot n \\log n)` where :math:`n` = samples,
      :math:`m` = features (sorting per feature per node)
    - Prediction: :math:`O(d)` per sample where :math:`d` = tree depth, fully
      vectorised across the batch

    **When to use DecisionTreeClassifier:**

    - Large datasets where JIT-compiled splitting provides speedups
    - Batch prediction on GPU/TPU backends
    - When you need an interpretable single-tree model with hardware acceleration

    References
    ----------
    .. [Breiman1984] Breiman, L., Friedman, J., Olshen, R. and Stone, C. (1984).
           **Classification and Regression Trees.**
           *Wadsworth International Group*.
    See Also
    --------
    :class:`~tuiml.algorithms.trees.DecisionTreeRegressor` : CART regression tree.
    :class:`~tuiml.algorithms.trees.C45TreeClassifier` : C4.5 classifier with gain-ratio splitting.
    :class:`~tuiml.algorithms.trees.RandomForestClassifier` : Ensemble of random trees.

    Examples
    --------
    >>> from tuiml.algorithms.trees import DecisionTreeClassifier
    >>> import numpy as np
    >>>
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]])
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>>
    >>> clf = DecisionTreeClassifier(max_depth=3)
    >>> clf.fit(X, y)
    DecisionTreeClassifier(max_depth=3, n_nodes=...)
    >>> predictions = clf.predict(X)
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        random_state: Optional[int] = None,
    ):
        """Initialize DecisionTreeClassifier.

        Parameters
        ----------
        criterion : str, default="gini"
            Splitting criterion: ``"gini"``, ``"entropy"``,
            ``"log_loss"`` (alias for entropy), or ``"gain_ratio"``
            (C4.5).
        max_depth : int or None, default=None
            Maximum tree depth.
        min_samples_split : int, default=2
            Minimum samples to split a node.
        min_samples_leaf : int, default=1
            Minimum samples at a leaf.
        min_impurity_decrease : float, default=0.0
            Minimum impurity decrease for a split.
        ccp_alpha : float, default=0.0
            Complexity parameter for pruning.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

        self.tree_ = None
        self.flat_tree_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.max_depth_ = None
        self.n_nodes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "criterion": {
                "type": "string",
                "default": "gini",
                "enum": ["gini", "entropy", "log_loss", "gain_ratio"],
                "description": "Splitting criterion: 'gini', 'entropy', 'log_loss', or 'gain_ratio'",
            },
            "max_depth": {
                "type": ["integer", "null"],
                "default": None,
                "minimum": 1,
                "description": "Maximum depth of the tree (None = unlimited)",
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "minimum": 2,
                "description": "Minimum samples required to split a node",
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Minimum samples required at a leaf node",
            },
            "min_impurity_decrease": {
                "type": "number",
                "default": 0.0,
                "minimum": 0.0,
                "description": "Minimum impurity decrease for a split to occur",
            },
            "ccp_alpha": {
                "type": "number",
                "default": 0.0,
                "minimum": 0.0,
                "description": "Complexity parameter for cost-complexity pruning",
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
        return ["numeric", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return (
            "Training: O(n * m * n*log(n)), "
            "Prediction: O(d) per sample, "
            "where n=samples, m=features, d=tree depth"
        )

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Breiman, L., Friedman, J., Olshen, R. and Stone, C. (1984). "
            "Classification and Regression Trees. Wadsworth.",
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Fit the CART classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator.
        """

        valid_criteria = ("gini", "entropy", "log_loss", "gain_ratio")
        if self.criterion not in valid_criteria:
            raise ValueError(
                f"Invalid criterion '{self.criterion}'. "
                f"Supported criteria: {valid_criteria}"
            )

        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        class_map = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([class_map[c] for c in y], dtype=np.intp)

        self.n_features_ = X.shape[1]
        rng = np.random.RandomState(self.random_state)

        config = TreeConfig(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            criterion=self.criterion,
            n_classes=self.n_classes_,
        )

        self.tree_ = build_classifier_tree(X, y_encoded, config, rng)

        if self.ccp_alpha > 0.0:
            self.tree_ = cost_complexity_prune(self.tree_, self.ccp_alpha)

        self.flat_tree_ = flatten_tree(self.tree_, value_width=self.n_classes_)
        self.n_nodes_ = self.flat_tree_.n_nodes
        self.max_depth_ = max_depth_of(self.tree_)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in *X*.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in *X*.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probability estimates.
        """
        self._check_is_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        return predict_proba_batch(self.flat_tree_, X)

    def get_tree_description(self, node=None, depth: int = 0) -> str:
        """Return a human-readable text description of the tree.

        Parameters
        ----------
        node : TreeNode or None
            Starting node (defaults to ``tree_``).
        depth : int
            Current indentation depth.

        Returns
        -------
        desc : str
            Multi-line text representation.
        """
        if node is None:
            node = self.tree_
        return get_tree_description(node, self.classes_, depth, "classification")

    def __repr__(self) -> str:
        """Return string representation.

        Returns
        -------
        repr : str
            String representation.
        """
        if self._is_fitted:
            return (
                f"DecisionTreeClassifier(max_depth={self.max_depth}, "
                f"n_nodes={self.n_nodes_})"
            )
        return f"DecisionTreeClassifier(max_depth={self.max_depth})"


# =========================================================================
# DecisionTreeRegressor
# =========================================================================

@regressor(tags=["trees", "interpretable"], version="1.0.0")
class DecisionTreeRegressor(Regressor):
    """CART decision tree for regression.

    A **CART** regression tree with configurable splitting criteria
    (squared error, Friedman MSE, or absolute error) with vectorised batch prediction.

    Overview
    --------
    1. Build the tree recursively using the selected **impurity criterion**
       (squared error, Friedman MSE, or absolute error) to evaluate splits
    2. After fitting, **flatten** the tree into parallel arrays
    3. Predict via efficient tree traversal

    Theory
    ------
    **Squared error** (``criterion="squared_error"``):

    .. math::
        \\text{MSE}(S) = \\frac{1}{|S|} \\sum_{i=1}^{|S|} (y_i - \\bar{y})^2

    **Friedman MSE** (``criterion="friedman_mse"``):

    .. math::
        \\Delta_{\\text{friedman}} = \\frac{n_L \\cdot n_R}{n^2}
        (\\bar{y}_L - \\bar{y}_R)^2

    **Absolute error** (``criterion="absolute_error"``):

    .. math::
        \\text{MAE}(S) = \\frac{1}{|S|} \\sum_{i=1}^{|S|}
        |y_i - \\text{median}(y)|

    Each leaf stores the **mean** (squared error, Friedman MSE) or
    **median** (absolute error) of its training targets.

    Parameters
    ----------
    criterion : str, default="squared_error"
        The function to measure the quality of a split. Supported criteria:
        ``"squared_error"`` for variance reduction (CART), ``"friedman_mse"``
        for Friedman's improvement score (better for boosting), and
        ``"absolute_error"`` for mean absolute error using median
        predictions.
    max_depth : int or None, default=None
        Maximum depth of the tree. ``None`` means unlimited.
    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node.
    min_impurity_decrease : float, default=0.0
        A node is split only if the impurity decrease is at least this value.
    ccp_alpha : float, default=0.0
        Complexity parameter for minimal cost-complexity pruning.
    random_state : int or None, default=None
        Random seed for reproducibility when features are tied.

    Attributes
    ----------
    tree_ : TreeNode
        The root node of the recursive tree (training representation).
    flat_tree_ : FlattenedTree
        Flattened parallel-array tree used for JIT prediction.
    n_features_ : int
        Number of features seen during ``fit()``.
    max_depth_ : int
        Actual depth of the fitted tree.
    n_nodes_ : int
        Total number of nodes in the fitted tree.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot m \\cdot n \\log n)` where :math:`n` = samples,
      :math:`m` = features
    - Prediction: :math:`O(d)` per sample where :math:`d` = tree depth, fully
      vectorised across the batch

    **When to use DecisionTreeRegressor:**

    - Large regression datasets where JIT-compiled criteria speed up training
    - Batch prediction on GPU/TPU backends
    - When you need an interpretable single-tree regression model with hardware
      acceleration

    References
    ----------
    .. [Breiman1984] Breiman, L., Friedman, J., Olshen, R. and Stone, C. (1984).
           **Classification and Regression Trees.**
           *Wadsworth International Group*.
    See Also
    --------
    :class:`~tuiml.algorithms.trees.DecisionTreeClassifier` : CART classifier.
    :class:`~tuiml.algorithms.trees.C45TreeRegressor` : C4.5-style regression tree.
    :class:`~tuiml.algorithms.trees.RandomForestRegressor` : Ensemble of random regression trees.

    Examples
    --------
    >>> from tuiml.algorithms.trees import DecisionTreeRegressor
    >>> import numpy as np
    >>>
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1.0, 2.0, 3.0, 4.0])
    >>>
    >>> reg = DecisionTreeRegressor(max_depth=3)
    >>> reg.fit(X, y)
    DecisionTreeRegressor(max_depth=3, n_nodes=...)
    >>> predictions = reg.predict(X)
    """

    def __init__(
        self,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        random_state: Optional[int] = None,
    ):
        """Initialize DecisionTreeRegressor.

        Parameters
        ----------
        criterion : str, default="squared_error"
            Splitting criterion: ``"squared_error"``, ``"friedman_mse"``,
            or ``"absolute_error"``.
        max_depth : int or None, default=None
            Maximum tree depth.
        min_samples_split : int, default=2
            Minimum samples to split a node.
        min_samples_leaf : int, default=1
            Minimum samples at a leaf.
        min_impurity_decrease : float, default=0.0
            Minimum impurity decrease for a split.
        ccp_alpha : float, default=0.0
            Complexity parameter for pruning.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

        self.tree_ = None
        self.flat_tree_ = None
        self.n_features_ = None
        self.max_depth_ = None
        self.n_nodes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "criterion": {
                "type": "string",
                "default": "squared_error",
                "enum": ["squared_error", "friedman_mse", "absolute_error"],
                "description": "Splitting criterion: 'squared_error', 'friedman_mse', or 'absolute_error'",
            },
            "max_depth": {
                "type": ["integer", "null"],
                "default": None,
                "minimum": 1,
                "description": "Maximum depth of the tree (None = unlimited)",
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "minimum": 2,
                "description": "Minimum samples required to split a node",
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Minimum samples required at a leaf node",
            },
            "min_impurity_decrease": {
                "type": "number",
                "default": 0.0,
                "minimum": 0.0,
                "description": "Minimum impurity decrease for a split to occur",
            },
            "ccp_alpha": {
                "type": "number",
                "default": 0.0,
                "minimum": 0.0,
                "description": "Complexity parameter for cost-complexity pruning",
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
        return ["numeric", "regression"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return (
            "Training: O(n * m * n*log(n)), "
            "Prediction: O(d) per sample, "
            "where n=samples, m=features, d=tree depth"
        )

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Breiman, L., Friedman, J., Olshen, R. and Stone, C. (1984). "
            "Classification and Regression Trees. Wadsworth.",
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        """Fit the CART regressor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : DecisionTreeRegressor
            Fitted estimator.
        """

        valid_criteria = ("squared_error", "friedman_mse", "absolute_error")
        if self.criterion not in valid_criteria:
            raise ValueError(
                f"Invalid criterion '{self.criterion}'. "
                f"Supported criteria: {valid_criteria}"
            )

        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()

        self.n_features_ = X.shape[1]
        rng = np.random.RandomState(self.random_state)

        config = TreeConfig(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            criterion=self.criterion,
        )

        self.tree_ = build_regressor_tree(X, y, config, rng)

        if self.ccp_alpha > 0.0:
            self.tree_ = cost_complexity_prune(self.tree_, self.ccp_alpha)

        self.flat_tree_ = flatten_tree(self.tree_, value_width=1)
        self.n_nodes_ = self.flat_tree_.n_nodes
        self.max_depth_ = max_depth_of(self.tree_)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples in *X*.

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
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        raw = predict_batch(self.flat_tree_, X)
        return raw.ravel()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the R-squared score on the given test data.

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
        self._check_is_fitted()
        y = np.asarray(y, dtype=np.float64).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0.0:
            return 0.0
        return 1.0 - ss_res / ss_tot

    def get_tree_description(self, node=None, depth: int = 0) -> str:
        """Return a human-readable text description of the tree.

        Parameters
        ----------
        node : TreeNode or None
            Starting node (defaults to ``tree_``).
        depth : int
            Current indentation depth.

        Returns
        -------
        desc : str
            Multi-line text representation.
        """
        if node is None:
            node = self.tree_
        return get_tree_description(node, depth=depth, task="regression")

    def __repr__(self) -> str:
        """Return string representation.

        Returns
        -------
        repr : str
            String representation.
        """
        if self._is_fitted:
            return (
                f"DecisionTreeRegressor(max_depth={self.max_depth}, "
                f"n_nodes={self.n_nodes_})"
            )
        return f"DecisionTreeRegressor(max_depth={self.max_depth})"

