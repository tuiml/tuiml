"""M5 Model Tree implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from tuiml.base.algorithms import Regressor, regressor
from ._core import TreeNode

# Backward compatibility alias
M5ModelTreeRegressorNode = TreeNode

@regressor(tags=["trees", "regression", "model-tree", "interpretable"], version="1.0.0")
class M5ModelTreeRegressor(Regressor):
    """M5 Model Tree for regression.

    Model trees are decision trees with **linear regression functions** at
    the leaves. This combines the **interpretability** of decision trees with
    the accuracy of **linear models**, producing piecewise-linear
    approximations of the target function.

    Overview
    --------
    The M5 algorithm builds a model tree in three stages:

    1. **Tree growing:** Recursively split the data using **Standard
       Deviation Reduction (SDR)** as the splitting criterion
    2. **Pruning:** Apply reduced-error pruning, replacing subtrees with
       linear models when doing so reduces the adjusted error
    3. **Smoothing:** Optionally smooth predictions by combining leaf
       predictions with parent node predictions to reduce discontinuities

    At each leaf, a **multiple linear regression** model is fitted to the
    subset of training data that reaches that leaf.

    Theory
    ------
    The Standard Deviation Reduction for splitting set :math:`T` on
    attribute :math:`A` at threshold :math:`t` is:

    .. math::
        SDR(T, A, t) = \\sigma(T) - \\frac{|T_L|}{|T|} \\sigma(T_L)
        - \\frac{|T_R|}{|T|} \\sigma(T_R)

    where :math:`\\sigma(\\cdot)` is the standard deviation of the target values.

    The **smoothed prediction** at a leaf with :math:`n` samples is:

    .. math::
        p' = \\frac{n \\cdot p_{leaf} + k \\cdot p_{parent}}{n + k}

    where :math:`k = 15` is the smoothing constant, :math:`p_{leaf}` is the
    leaf model prediction, and :math:`p_{parent}` is the parent node
    prediction.

    The linear model at each leaf solves:

    .. math::
        \\hat{y} = X \\beta + \\beta_0

    using **ridge-regularized least squares**.

    Parameters
    ----------
    min_samples_leaf : int, default=4
        Minimum samples at a leaf node.
    unpruned : bool, default=False
        If True, build an unpruned tree.
    use_unsmoothed : bool, default=False
        If True, use unsmoothed predictions.
    build_regression_tree : bool, default=False
        If True, build a regression tree with constant predictions
        at leaves instead of linear models.

    Attributes
    ----------
    tree_ : TreeNode
        The fitted model tree.
    n_features_ : int
        Number of features seen during fit.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot m \\cdot \\log(n))` where :math:`n` = samples,
      :math:`m` = features, plus :math:`O(n \\cdot m^2)` for linear model
      fitting at each leaf
    - Prediction: :math:`O(\\log(n) + m)` per sample (tree traversal plus
      linear model evaluation)

    **When to use M5ModelTreeRegressor:**

    - When the target function has **piecewise-linear** structure
    - When you need an **interpretable** regression model
    - When pure regression trees (constant leaves) are too coarse
    - When a global linear model is too simple for the data
    - Tabular data with moderate to high dimensionality

    References
    ----------
    .. [Quinlan1992] Quinlan, J.R. (1992).
           **Learning with Continuous Classes.**
           *Proceedings of the 5th Australian Joint Conference on Artificial
           Intelligence*, pp. 343-348.

    .. [Wang1997] Wang, Y. and Witten, I.H. (1997).
           **Induction of Model Trees for Predicting Continuous Classes.**
           *Proceedings of the European Conference on Machine Learning (ECML)*.

    See Also
    --------
    :class:`~tuiml.algorithms.trees.C45TreeClassifier` : Classification tree using gain ratio.
    :class:`~tuiml.algorithms.trees.ReducedErrorPruningTreeClassifier` : Classification tree with REP pruning.
    :class:`~tuiml.algorithms.trees.LogisticModelTreeClassifier` : Classification tree with logistic models at leaves.

    Examples
    --------
    Basic usage for regression:

    >>> from tuiml.algorithms.trees import M5ModelTreeRegressor
    >>> import numpy as np
    >>>
    >>> # Create sample regression data
    >>> X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    >>> y = np.array([1.1, 2.0, 3.1, 4.0, 5.2, 6.1, 7.0, 8.1])
    >>>
    >>> # Fit a model tree
    >>> reg = M5ModelTreeRegressor(min_samples_leaf=4)
    >>> reg.fit(X, y)
    M5ModelTreeRegressor(...)
    >>> predictions = reg.predict(X)
    """

    def __init__(
        self,
        min_samples_leaf: int = 4,
        unpruned: bool = False,
        use_unsmoothed: bool = False,
        build_regression_tree: bool = False,
    ):
        """Initialize M5ModelTreeRegressor.

        Parameters
        ----------
        min_samples_leaf : int, default=4
            Minimum samples per leaf.
        unpruned : bool, default=False
            Whether to build unpruned tree.
        use_unsmoothed : bool, default=False
            Use unsmoothed predictions.
        build_regression_tree : bool, default=False
            Build regression tree (constant at leaves).
        """
        super().__init__()
        self.min_samples_leaf = min_samples_leaf
        self.unpruned = unpruned
        self.use_unsmoothed = use_unsmoothed
        self.build_regression_tree = build_regression_tree

        # Fitted attributes
        self.tree_ = None
        self.n_features_ = None
        self._global_std = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "min_samples_leaf": {
                "type": "integer",
                "default": 4,
                "minimum": 1,
                "description": "Minimum samples at leaf"
            },
            "unpruned": {
                "type": "boolean",
                "default": False,
                "description": "Build unpruned tree"
            },
            "use_unsmoothed": {
                "type": "boolean",
                "default": False,
                "description": "Use unsmoothed predictions"
            },
            "build_regression_tree": {
                "type": "boolean",
                "default": False,
                "description": "Build regression tree instead of model tree"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return [
            "numeric",
            "missing_values",
            "numeric_class"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m * log(n)) training, O(log(n)) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Quinlan, J.R. (1992). Learning with Continuous Classes. "
            "5th Australian Joint Conference on AI, 343-348.",
            "Wang, Y. & Witten, I.H. (1997). Induction of model trees for "
            "predicting continuous classes."
        ]

    def _compute_std(self, y: np.ndarray) -> float:
        """Compute standard deviation of target values.

        Parameters
        ----------
        y : np.ndarray
            Target values.

        Returns
        -------
        std : float
            Sample standard deviation (ddof=1).
        """
        if len(y) < 2:
            return 0.0
        return np.std(y, ddof=1)

    def _compute_sdr(self, y: np.ndarray, y_left: np.ndarray,
                     y_right: np.ndarray) -> float:
        """Compute Standard Deviation Reduction (SDR).

        Parameters
        ----------
        y : np.ndarray
            Parent node target values.
        y_left : np.ndarray
            Left child target values.
        y_right : np.ndarray
            Right child target values.

        Returns
        -------
        sdr : float
            Standard deviation reduction.
        """
        n = len(y)
        if n == 0 or len(y_left) == 0 or len(y_right) == 0:
            return 0.0
        std_total = self._compute_std(y)
        std_left = self._compute_std(y_left)
        std_right = self._compute_std(y_right)
        return std_total - (len(y_left) / n) * std_left - (len(y_right) / n) * std_right

    def _find_best_split(self, X: np.ndarray, y: np.ndarray
                         ) -> Tuple[int, float, float]:
        """Find the best split using incremental statistics.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        best_attr : int
            Index of the best splitting attribute.
        best_value : float
            Best split threshold.
        best_sdr : float
            Standard deviation reduction of the best split.
        """
        n_samples, n_features = X.shape
        best_sdr = -float('inf')
        best_attr = -1
        best_value = 0.0

        # Precompute total statistics (constant for all splits in this node)
        total_std = self._compute_std(y)
        total_sum = np.sum(y)
        total_sq_sum = np.sum(y ** 2)
        n = n_samples

        for attr in range(n_features):
            # Sort once per feature
            order = np.argsort(X[:, attr])
            x_sorted = X[order, attr]
            y_sorted = y[order]

            # Skip NaN values at the end (argsort puts NaN last)
            valid_n = n
            while valid_n > 0 and np.isnan(x_sorted[valid_n - 1]):
                valid_n -= 1
            if valid_n < 2 * self.min_samples_leaf:
                continue

            # Incremental statistics
            left_sum = 0.0
            left_sq_sum = 0.0

            for i in range(valid_n - 1):
                left_sum += y_sorted[i]
                left_sq_sum += y_sorted[i] ** 2
                n_left = i + 1
                n_right = valid_n - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                # Skip duplicate feature values
                if x_sorted[i] == x_sorted[i + 1]:
                    continue

                right_sum = total_sum - left_sum
                right_sq_sum = total_sq_sum - left_sq_sum

                # Incremental std: sqrt((sum_sq - sum^2/n) / (n-1))
                if n_left > 1:
                    left_var = (left_sq_sum - left_sum ** 2 / n_left) / (n_left - 1)
                    left_std = np.sqrt(max(0.0, left_var))
                else:
                    left_std = 0.0

                if n_right > 1:
                    right_var = (right_sq_sum - right_sum ** 2 / n_right) / (n_right - 1)
                    right_std = np.sqrt(max(0.0, right_var))
                else:
                    right_std = 0.0

                sdr = total_std - (n_left / n) * left_std - (n_right / n) * right_std

                if sdr > best_sdr:
                    best_sdr = sdr
                    best_attr = attr
                    best_value = (x_sorted[i] + x_sorted[i + 1]) / 2.0

        return best_attr, best_value, best_sdr

    def _fit_linear_model(self, X: np.ndarray, y: np.ndarray
                          ) -> Tuple[np.ndarray, float]:
        """Fit a linear model to the data at a leaf.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix at the leaf.
        y : np.ndarray of shape (n_samples,)
            Target values at the leaf.

        Returns
        -------
        coefficients : np.ndarray of shape (n_features,)
            Regression coefficients.
        intercept : float
            Intercept term.
        """
        n_samples, n_features = X.shape

        if n_samples < n_features + 1:
            return np.zeros(n_features), np.mean(y)

        X_bias = np.column_stack([np.ones(n_samples), X])

        try:
            ridge = 1.0
            XtX = X_bias.T @ X_bias + ridge * np.eye(n_features + 1)
            Xty = X_bias.T @ y
            coeffs = np.linalg.solve(XtX, Xty)
            intercept = coeffs[0]
            coeffs = coeffs[1:]
        except np.linalg.LinAlgError:
            return np.zeros(n_features), np.mean(y)

        return coeffs, intercept

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        """Recursively build the model tree.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for the current node.
        y : np.ndarray of shape (n_samples,)
            Target values.
        depth : int, default=0
            Current depth in the tree.

        Returns
        -------
        node : TreeNode
            Root of the constructed subtree.
        """
        n_samples = len(y)
        node = TreeNode(n_samples=n_samples)

        # Always compute linear model/prediction for smoothing
        node.predicted_value = float(np.mean(y))
        node.value = np.array([node.predicted_value])
        if not self.build_regression_tree and n_samples >= self.n_features_ + 1:
            coeffs, intercept = self._fit_linear_model(X, y)
            node.linear_model = coeffs
            node.intercept = intercept

        # Check stopping conditions
        if n_samples < 2 * self.min_samples_leaf:
            node.is_leaf = True
            return node

        std = self._compute_std(y)
        if std < 0.005 * self._global_std:
            node.is_leaf = True
            return node

        best_attr, best_value, best_sdr = self._find_best_split(X, y)

        if best_attr < 0 or best_sdr <= 0:
            node.is_leaf = True
            return node

        node.feature_index = best_attr
        node.threshold = best_value

        left_mask = X[:, best_attr] <= best_value
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return node

    def _prune_tree(self, node: TreeNode, X: np.ndarray, y: np.ndarray) -> TreeNode:
        """Prune the tree using reduced error pruning.

        Parameters
        ----------
        node : TreeNode
            Current node to evaluate for pruning.
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        node : TreeNode
            Pruned node (leaf or original subtree).
        """
        if node.is_leaf:
            return node

        left_mask = X[:, node.feature_index] <= node.threshold
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]

        if len(y_left) > 0:
            node.left = self._prune_tree(node.left, X_left, y_left)
        if len(y_right) > 0:
            node.right = self._prune_tree(node.right, X_right, y_right)

        # Subtree error (vectorized batch prediction)
        subtree_preds = self._predict_batch(node, X)
        subtree_error = np.sum((y - subtree_preds) ** 2)

        # Leaf error
        if self.build_regression_tree:
            leaf_pred = np.mean(y)
            leaf_error = np.sum((y - leaf_pred) ** 2)
            coeffs, intercept = None, 0.0
        else:
            coeffs, intercept = self._fit_linear_model(X, y)
            leaf_pred = X @ coeffs + intercept
            leaf_error = np.sum((y - leaf_pred) ** 2)

        n_params = self.n_features_ + 1 if not self.build_regression_tree else 1
        n = len(y)
        adjusted_error = leaf_error * (n + n_params) / (n - n_params) if n > n_params else leaf_error

        if adjusted_error <= subtree_error:
            new_node = TreeNode(is_leaf=True, n_samples=n)
            new_node.predicted_value = float(np.mean(y))
            new_node.value = np.array([new_node.predicted_value])
            if not self.build_regression_tree and coeffs is not None:
                new_node.linear_model = coeffs
                new_node.intercept = intercept
            return new_node

        return node

    def _predict_batch(self, node: TreeNode, X: np.ndarray) -> np.ndarray:
        """Batch predict for all samples through a subtree.

        Parameters
        ----------
        node : TreeNode
            Root of the subtree.
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        if node.is_leaf:
            if self.build_regression_tree or node.linear_model is None:
                return np.full(len(X), node.predicted_value)
            return X @ node.linear_model + node.intercept

        n = len(X)
        predictions = np.zeros(n)
        mask = X[:, node.feature_index] <= node.threshold

        # Handle NaN
        nan_mask = np.isnan(X[:, node.feature_index])
        if np.any(nan_mask):
            if node.left.n_samples >= node.right.n_samples:
                mask[nan_mask] = True
            else:
                mask[nan_mask] = False

        left_idx = np.where(mask)[0]
        right_idx = np.where(~mask)[0]

        if len(left_idx) > 0:
            predictions[left_idx] = self._predict_batch(node.left, X[left_idx])
        if len(right_idx) > 0:
            predictions[right_idx] = self._predict_batch(node.right, X[right_idx])

        return predictions

    def _predict_batch_smooth(self, node: TreeNode, X: np.ndarray,
                               parent_pred: Optional[np.ndarray] = None
                               ) -> np.ndarray:
        """Batch predict with smoothing for all samples.

        Parameters
        ----------
        node : TreeNode
            Current tree node.
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        parent_pred : np.ndarray of shape (n_samples,), optional
            Parent node predictions for smoothing.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Smoothed predicted values.
        """
        n = len(X)

        # Compute current node prediction
        if self.build_regression_tree or node.linear_model is None:
            current_pred = np.full(n, node.predicted_value)
        else:
            current_pred = X @ node.linear_model + node.intercept

        if node.is_leaf:
            if parent_pred is not None:
                k = 15  # Smoothing constant
                ns = node.n_samples
                return (ns * current_pred + k * parent_pred) / (ns + k)
            return current_pred

        predictions = np.zeros(n)
        mask = X[:, node.feature_index] <= node.threshold

        # Handle NaN
        nan_mask = np.isnan(X[:, node.feature_index])
        if np.any(nan_mask):
            if node.left.n_samples >= node.right.n_samples:
                mask[nan_mask] = True
            else:
                mask[nan_mask] = False

        left_idx = np.where(mask)[0]
        right_idx = np.where(~mask)[0]

        if len(left_idx) > 0:
            predictions[left_idx] = self._predict_batch_smooth(
                node.left, X[left_idx], current_pred[left_idx])
        if len(right_idx) > 0:
            predictions[right_idx] = self._predict_batch_smooth(
                node.right, X[right_idx], current_pred[right_idx])

        return predictions

    def _predict_single(self, node: TreeNode, x: np.ndarray) -> float:
        """Predict for a single sample.

        Parameters
        ----------
        node : TreeNode
            Current tree node.
        x : np.ndarray of shape (n_features,)
            Single sample feature vector.

        Returns
        -------
        prediction : float
            Predicted target value.
        """
        if node.is_leaf:
            if self.build_regression_tree or node.linear_model is None:
                return node.predicted_value
            return float(x @ node.linear_model + node.intercept)

        if np.isnan(x[node.feature_index]):
            if node.left.n_samples >= node.right.n_samples:
                return self._predict_single(node.left, x)
            return self._predict_single(node.right, x)

        if x[node.feature_index] <= node.threshold:
            return self._predict_single(node.left, x)
        return self._predict_single(node.right, x)

    def _smooth_predictions(self, node: TreeNode, x: np.ndarray,
                            parent_pred: float = None) -> float:
        """Apply smoothing to predictions.

        Parameters
        ----------
        node : TreeNode
            Current tree node.
        x : np.ndarray of shape (n_features,)
            Single sample feature vector.
        parent_pred : float or None, default=None
            Prediction from the parent node for smoothing.

        Returns
        -------
        prediction : float
            Smoothed predicted target value.
        """
        if node.is_leaf:
            if self.build_regression_tree or node.linear_model is None:
                leaf_pred = node.predicted_value
            else:
                leaf_pred = float(x @ node.linear_model + node.intercept)

            if parent_pred is not None and not self.use_unsmoothed:
                k = 15  # Smoothing constant
                n = node.n_samples
                return (n * leaf_pred + k * parent_pred) / (n + k)
            return leaf_pred

        # Compute prediction at this node
        if self.build_regression_tree or node.linear_model is None:
            current_pred = node.predicted_value
        else:
            current_pred = float(x @ node.linear_model + node.intercept)

        if np.isnan(x[node.feature_index]):
            if node.left.n_samples >= node.right.n_samples:
                return self._smooth_predictions(node.left, x, current_pred)
            return self._smooth_predictions(node.right, x, current_pred)

        if x[node.feature_index] <= node.threshold:
            return self._smooth_predictions(node.left, x, current_pred)
        return self._smooth_predictions(node.right, x, current_pred)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "M5ModelTreeRegressor":
        """Fit the M5ModelTreeRegressor model tree.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : M5ModelTreeRegressor
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, self.n_features_ = X.shape
        self._global_std = self._compute_std(y)

        self.tree_ = self._build_tree(X, y)

        if not self.unpruned:
            self.tree_ = self._prune_tree(self.tree_, X, y)

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
            Predicted values.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.use_unsmoothed:
            return self._predict_batch(self.tree_, X)
        else:
            return self._predict_batch_smooth(self.tree_, X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R-squared score.

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
        self._check_is_fitted()
        y_pred = self.predict(X)
        y = np.asarray(y)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)

    def _count_leaves(self, node: TreeNode) -> int:
        """Count number of leaves in tree.

        Parameters
        ----------
        node : TreeNode
            Subtree root.

        Returns
        -------
        count : int
            Number of leaf nodes in the subtree.
        """
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            n_leaves = self._count_leaves(self.tree_)
            tree_type = "Regression tree" if self.build_regression_tree else "Model tree"
            return f"M5ModelTreeRegressor({tree_type}, n_leaves={n_leaves})"
        return f"M5ModelTreeRegressor(min_samples_leaf={self.min_samples_leaf})"
