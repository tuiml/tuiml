"""Logistic Model Tree classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

from tuiml.base.algorithms import Classifier, classifier
from ._core import TreeNode, entropy

# Backward compatibility alias
LogisticModelTreeClassifierNode = TreeNode

@classifier(tags=["trees", "logistic", "interpretable"], version="1.0.0")
class LogisticModelTreeClassifier(Classifier):
    """Logistic Model Trees classifier.

    LogisticModelTreeClassifier builds a decision tree with **logistic
    regression functions** at the leaves. It uses the **LogitBoost**
    algorithm to fit the logistic models and applies the **C4.5 information
    gain** criterion for tree splitting.

    Overview
    --------
    The algorithm builds a tree with logistic models at each node:

    1. At each node, fit a **logistic regression model** using a simplified
       LogitBoost procedure with iterative Newton steps
    2. Evaluate candidate splits using **information gain** (C4.5 criterion)
    3. Select the best feature and threshold for splitting
    4. Recursively build child subtrees, inheriting the parent's logistic
       model as a starting point
    5. At prediction time, traverse the tree to a leaf and use that leaf's
       **logistic model** for class probability estimation

    Theory
    ------
    The logistic model at each node uses the **sigmoid function** for binary
    classification:

    .. math::
        P(y=1 | x) = \\sigma(w^T x + b) = \\frac{1}{1 + e^{-(w^T x + b)}}

    The LogitBoost update for weight :math:`w_j` at each iteration is:

    .. math::
        w_j \\leftarrow w_j + \\eta \\cdot \\frac{\\sum_i (y_i - p_i) x_{ij}}{\\sum_i p_i(1 - p_i) + \\epsilon}

    where :math:`\\eta` is the learning rate, :math:`p_i = \\sigma(w^T x_i + b)`,
    and :math:`\\epsilon` is a small constant for numerical stability.

    For multiclass problems, a **one-vs-all** scheme is used with separate
    binary logistic regressors for each class.

    The tree splitting uses information gain:

    .. math::
        IG(S, A) = H(S) - \\sum_{v} \\frac{|S_v|}{|S|} H(S_v)

    Parameters
    ----------
    min_samples_leaf : int, default=15
        Minimum samples at a leaf node.
    max_depth : int, optional
        Maximum tree depth. None means unlimited.
    num_boosting_iterations : int, default=-1
        Number of LogitBoost iterations (-1 for auto).
    use_aic : bool, default=False
        Use AIC for model selection stopping criterion.
    learning_rate : float, default=0.5
        Learning rate for LogitBoost.

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

    - Training: :math:`O(n \\cdot m \\cdot \\log(n) \\cdot B)` where :math:`n` =
      samples, :math:`m` = features, :math:`B` = boosting iterations
    - Prediction: :math:`O(\\log(n) \\cdot m)` per sample (tree traversal plus
      logistic model evaluation)

    **When to use LogisticModelTreeClassifier:**

    - When you need a model that is more accurate than a plain decision tree
      but more **interpretable** than a black-box ensemble
    - When decision boundaries are **locally linear** in feature space
    - Binary or multiclass classification with numeric features
    - When you want the **structure** of a tree with the **smoothness** of
      logistic regression at the leaves

    References
    ----------
    .. [Landwehr2005] Landwehr, N., Hall, M. and Frank, E. (2005).
           **Logistic Model Trees.**
           *Machine Learning*, 59(1-2), pp. 161-205.
           DOI: `10.1007/s10994-005-0466-3 <https://doi.org/10.1007/s10994-005-0466-3>`_

    .. [Friedman2000] Friedman, J., Hastie, T. and Tibshirani, R. (2000).
           **Additive Logistic Regression: A Statistical View of Boosting.**
           *The Annals of Statistics*, 28(2), pp. 337-407.
           DOI: `10.1214/aos/1016218223 <https://doi.org/10.1214/aos/1016218223>`_

    See Also
    --------
    :class:`~tuiml.algorithms.trees.C45TreeClassifier` : Classification tree using gain ratio.
    :class:`~tuiml.algorithms.trees.M5ModelTreeRegressor` : Regression tree with linear models at leaves.
    :class:`~tuiml.algorithms.trees.RandomForestClassifier` : Ensemble of randomized trees.

    Examples
    --------
    Basic usage for classification:

    >>> from tuiml.algorithms.trees import LogisticModelTreeClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]])
    >>> y = np.array([0, 0, 1, 1, 0, 1])
    >>>
    >>> # Fit a logistic model tree
    >>> clf = LogisticModelTreeClassifier(min_samples_leaf=15)
    >>> clf.fit(X, y)
    LogisticModelTreeClassifier(...)
    >>> predictions = clf.predict(X)
    """

    def __init__(self, min_samples_leaf: int = 15,
                 max_depth: Optional[int] = None,
                 num_boosting_iterations: int = -1,
                 use_aic: bool = False,
                 learning_rate: float = 0.5):
        super().__init__()
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.num_boosting_iterations = num_boosting_iterations
        self.use_aic = use_aic
        self.learning_rate = learning_rate
        self.tree_ = None
        self.classes_ = None
        self.n_features_ = None
        self._n_classes = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "min_samples_leaf": {
                "type": "integer", "default": 15, "minimum": 1,
                "description": "Minimum samples at leaf node"
            },
            "max_depth": {
                "type": "integer", "default": None, "minimum": 1,
                "description": "Maximum tree depth"
            },
            "num_boosting_iterations": {
                "type": "integer", "default": -1,
                "description": "Boosting iterations (-1 for auto)"
            },
            "use_aic": {
                "type": "boolean", "default": False,
                "description": "Use AIC for model selection"
            },
            "learning_rate": {
                "type": "number", "default": 0.5, "minimum": 0.01, "maximum": 1.0,
                "description": "Learning rate for LogitBoost"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(n * m * log(n) * B) where B is boosting iterations"

    @classmethod
    def get_references(cls) -> List[str]:
        return [
            "Landwehr, N., Hall, M., & Frank, E. (2005). Logistic Model Trees. "
            "Machine Learning, 59(1-2), 161-205."
        ]

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function.

        Parameters
        ----------
        x : np.ndarray
            Input logit values.

        Returns
        -------
        p : np.ndarray
            Sigmoid probabilities in [0, 1].
        """
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def _fit_logistic(self, X: np.ndarray, y: np.ndarray,
                      max_iter: int = 50) -> Tuple[np.ndarray, Any]:
        """Fit logistic regression using simplified LogitBoost.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Class labels (integer-encoded).
        max_iter : int, default=50
            Maximum number of boosting iterations.

        Returns
        -------
        weights : np.ndarray
            Logistic model weights.
        bias : float or np.ndarray
            Logistic model bias term(s).
        """
        n_samples, n_features = X.shape

        if self._n_classes == 2:
            weights = np.zeros(n_features)
            bias = 0.0

            for _ in range(max_iter):
                logits = X @ weights + bias
                probs = self._sigmoid(logits)
                residuals = y - probs
                working_weights = probs * (1 - probs) + 1e-6

                for j in range(n_features):
                    weights[j] += self.learning_rate * np.sum(
                        residuals * X[:, j]
                    ) / (np.sum(working_weights) + 1e-6)

                bias += self.learning_rate * np.sum(residuals) / (n_samples + 1e-6)

            return weights, bias
        else:
            weights = np.zeros((self._n_classes, n_features))
            bias = np.zeros(self._n_classes)

            for c in range(self._n_classes):
                y_bin = (y == c).astype(float)
                w, b = self._fit_logistic_binary(X, y_bin, max_iter)
                weights[c] = w
                bias[c] = b

            return weights, bias

    def _fit_logistic_binary(self, X: np.ndarray, y: np.ndarray,
                             max_iter: int) -> Tuple[np.ndarray, float]:
        """Fit binary logistic regression.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Binary target labels (0 or 1).
        max_iter : int
            Maximum number of boosting iterations.

        Returns
        -------
        weights : np.ndarray of shape (n_features,)
            Logistic model weights.
        bias : float
            Logistic model bias term.
        """
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0.0

        for _ in range(max_iter):
            logits = X @ weights + bias
            probs = self._sigmoid(logits)
            residuals = y - probs
            working_weights = probs * (1 - probs) + 1e-6

            for j in range(n_features):
                weights[j] += self.learning_rate * np.sum(
                    residuals * X[:, j]
                ) / (np.sum(working_weights) + 1e-6)

            bias += self.learning_rate * np.sum(residuals) / (n_samples + 1e-6)

        return weights, bias

    def _information_gain(self, y: np.ndarray, y_left: np.ndarray,
                          y_right: np.ndarray) -> float:
        """Calculate information gain using shared entropy.

        Parameters
        ----------
        y : np.ndarray
            Parent node labels.
        y_left : np.ndarray
            Left child labels.
        y_right : np.ndarray
            Right child labels.

        Returns
        -------
        ig : float
            Information gain value.
        """
        n = len(y)
        if n == 0:
            return 0.0
        parent_H = entropy(y, self._n_classes)
        n_left, n_right = len(y_left), len(y_right)
        child_H = (n_left / n * entropy(y_left, self._n_classes) +
                   n_right / n * entropy(y_right, self._n_classes))
        return parent_H - child_H

    def _find_best_split(self, X: np.ndarray, y: np.ndarray
                         ) -> Tuple[int, float, float]:
        """Find best split point.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Class labels.

        Returns
        -------
        best_feature : int
            Index of the best splitting feature.
        best_threshold : float
            Best split threshold.
        best_gain : float
            Information gain of the best split.
        """
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feature = 0
        best_threshold = 0

        for feature_idx in range(n_features):
            X_col = X[:, feature_idx]
            valid_mask = ~np.isnan(X_col)
            if np.sum(valid_mask) < 2 * self.min_samples_leaf:
                continue

            X_valid = X_col[valid_mask]
            y_valid = y[valid_mask]

            sorted_idx = np.argsort(X_valid)
            sorted_X = X_valid[sorted_idx]
            sorted_y = y_valid[sorted_idx]

            for i in range(len(sorted_X) - 1):
                if sorted_X[i] == sorted_X[i + 1]:
                    continue

                threshold = (sorted_X[i] + sorted_X[i + 1]) / 2
                y_left = sorted_y[:i + 1]
                y_right = sorted_y[i + 1:]

                if len(y_left) < self.min_samples_leaf or \
                   len(y_right) < self.min_samples_leaf:
                    continue

                gain = self._information_gain(y_valid, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        """Build the tree recursively.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for the current node.
        y : np.ndarray of shape (n_samples,)
            Class labels (integer-encoded).
        depth : int, default=0
            Current depth in the tree.

        Returns
        -------
        node : TreeNode
            Root of the constructed subtree.
        """
        n_samples = len(y)

        # Determine number of boosting iterations
        if self.num_boosting_iterations == -1:
            max_iter = min(200, n_samples // 5)
        else:
            max_iter = self.num_boosting_iterations
        max_iter = max(max_iter, 10)

        # Fit logistic regression at this node
        w, b = self._fit_logistic(X, y, max_iter)

        # Check stopping conditions
        if (n_samples < 2 * self.min_samples_leaf or
            len(set(y)) == 1 or
            (self.max_depth is not None and depth >= self.max_depth)):
            return TreeNode(
                is_leaf=True,
                weights=w,
                bias=b,
                n_samples=n_samples,
            )

        # Find best split
        feature_idx, threshold, gain = self._find_best_split(X, y)

        if gain <= 0:
            return TreeNode(
                is_leaf=True,
                weights=w,
                bias=b,
                n_samples=n_samples,
            )

        # Split data
        X_col = X[:, feature_idx]
        left_mask = X_col <= threshold
        right_mask = X_col > threshold

        # Handle missing values
        missing_mask = np.isnan(X_col)
        if np.any(missing_mask):
            if np.sum(left_mask) >= np.sum(right_mask):
                left_mask = left_mask | missing_mask
            else:
                right_mask = right_mask | missing_mask

        node = TreeNode(
            is_leaf=False,
            feature_index=feature_idx,
            threshold=threshold,
            weights=w,
            bias=b,
            n_samples=n_samples,
        )

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _predict_proba_node(self, node: TreeNode, x: np.ndarray) -> np.ndarray:
        """Get probability prediction from a node's logistic model.

        Parameters
        ----------
        node : TreeNode
            Node containing the logistic model.
        x : np.ndarray of shape (n_features,)
            Single sample feature vector.

        Returns
        -------
        proba : np.ndarray of shape (n_classes,)
            Class probability distribution.
        """
        if self._n_classes == 2:
            logit = x @ node.weights + node.bias
            p1 = self._sigmoid(logit)
            return np.array([1 - p1, p1])
        else:
            logits = node.weights @ x + node.bias
            exp_logits = np.exp(logits - np.max(logits))
            return exp_logits / (np.sum(exp_logits) + 1e-10)

    def _predict_proba_single(self, node: TreeNode, x: np.ndarray) -> np.ndarray:
        """Predict probabilities for a single sample.

        Parameters
        ----------
        node : TreeNode
            Current tree node.
        x : np.ndarray of shape (n_features,)
            Single sample feature vector.

        Returns
        -------
        proba : np.ndarray of shape (n_classes,)
            Class probability distribution.
        """
        if node.is_leaf:
            return self._predict_proba_node(node, x)

        value = x[node.feature_index]
        if np.isnan(value):
            left_proba = self._predict_proba_single(node.left, x)
            right_proba = self._predict_proba_single(node.right, x)
            total = node.left.n_samples + node.right.n_samples
            return (node.left.n_samples * left_proba +
                    node.right.n_samples * right_proba) / total
        elif value <= node.threshold:
            return self._predict_proba_single(node.left, x)
        else:
            return self._predict_proba_single(node.right, x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModelTreeClassifier":
        """Fit the LogisticModelTreeClassifier classifier."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self._n_classes = len(self.classes_)

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y])

        self.tree_ = self._build_tree(X, y_idx)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self._n_classes))

        for i in range(n_samples):
            proba[i] = self._predict_proba_single(self.tree_, X[i])

        return proba

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"LogisticModelTreeClassifier(n_features={self.n_features_}, classes={list(self.classes_)})"
        return f"LogisticModelTreeClassifier(min_samples_leaf={self.min_samples_leaf})"
