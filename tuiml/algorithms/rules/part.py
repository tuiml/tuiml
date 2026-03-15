"""PARTClassifier rule learner implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

from tuiml.base.algorithms import Classifier, classifier

@dataclass
class PartRule:
    """A rule derived from a partial decision tree."""
    conditions: List[Tuple[int, str, float]] = field(default_factory=list)
    predicted_class: Any = None
    n_correct: int = 0
    n_covered: int = 0

    def covers(self, x: np.ndarray) -> bool:
        """Check if rule covers an instance."""
        for feature_idx, operator, value in self.conditions:
            val = x[feature_idx]
            if np.isnan(val):
                continue
            if operator == '<=':
                if val > value:
                    return False
            elif operator == '>':
                if val <= value:
                    return False
        return True

    def __str__(self) -> str:
        if not self.conditions:
            return f"() => {self.predicted_class} ({self.n_correct}/{self.n_covered})"
        conds = " AND ".join([f"x[{f}] {op} {v:.4f}" for f, op, v in self.conditions])
        return f"({conds}) => {self.predicted_class} ({self.n_correct}/{self.n_covered})"

@classifier(tags=["rules", "partial-trees", "interpretable"], version="1.0.0")
class PARTClassifier(Classifier):
    """PART rule learner combining **partial decision trees** with
    **separate-and-conquer** rule extraction.

    PARTClassifier builds rules by repeatedly building partial C4.5 decision
    trees and extracting the best leaf as a rule. It combines the
    separate-and-conquer strategy with the decision tree approach, avoiding
    the need for global optimization.

    Overview
    --------
    The algorithm generates an ordered rule set as follows:

    1. While uncovered instances remain:
       a. Build a **partial** C4.5 decision tree on the remaining instances
          (expand only the most promising branch at each internal node)
       b. Extract the path to the leaf with highest purity as a new rule
       c. Remove all instances covered by the new rule
    2. Instances not covered by any rule are assigned the default (majority) class

    Theory
    ------
    At each split, the algorithm uses the **gain ratio** criterion to select
    the best feature and threshold:

    .. math::
        \\text{GainRatio}(A) = \\frac{H(S) - \\sum_{v} \\frac{|S_v|}{|S|} H(S_v)}{-\\sum_{v} \\frac{|S_v|}{|S|} \\log_2 \\frac{|S_v|}{|S|}}

    where :math:`H(S)` is the entropy of set :math:`S` and :math:`S_v` are
    the subsets after splitting on attribute :math:`A`:

    .. math::
        H(S) = -\\sum_{c \\in C} p_c \\log_2 p_c

    The partial tree strategy avoids fully expanding the tree, which reduces
    overfitting and computational cost compared to C4.5 followed by rule
    extraction.

    Parameters
    ----------
    min_samples_leaf : int, default=2
        Minimum number of samples allowed in a leaf.
    confidence_factor : float, default=0.25
        Confidence factor used for pruning.
    unpruned : bool, default=False
        Whether pruning should be performed.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    rules_ : list of PartRule
        List of learned classification rules.
    classes_ : np.ndarray
        Unique class labels.
    default_class_ : any
        Default class prediction for instances not covered by any rule.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n^2 \\cdot m \\cdot \\log(n))` where :math:`n` =
      number of samples and :math:`m` = number of features (partial tree
      construction repeated for each rule)
    - Prediction: :math:`O(r \\cdot m)` per sample where :math:`r` = number
      of rules

    **When to use PARTClassifier:**

    - When interpretable rules are needed without global optimization
    - Datasets where decision trees perform well (PART inherits C4.5 strengths)
    - When a balance between RIPPER and C4.5-based rules is desired
    - Multi-class classification problems with moderate dimensionality

    References
    ----------
    .. [Frank1998] Frank, E. and Witten, I.H. (1998).
           **Generating Accurate Rule Sets Without Global Optimization.**
           *Proceedings of the 15th International Conference on Machine Learning (ICML-98)*, pp. 144-151.

    See Also
    --------
    :class:`~tuiml.algorithms.rules.RIPPERClassifier` : RIPPER propositional rule learner.
    :class:`~tuiml.algorithms.rules.DecisionTableClassifier` : Decision table classifier.
    :class:`~tuiml.algorithms.rules.OneRuleClassifier` : Single-attribute rule learner.

    Examples
    --------
    Basic usage for rule-based classification via partial trees:

    >>> from tuiml.algorithms.rules import PARTClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    >>> y_train = np.array([0, 0, 1, 1, 1])
    >>>
    >>> # Fit the model
    >>> clf = PARTClassifier(min_samples_leaf=2)
    >>> clf.fit(X_train, y_train)
    PARTClassifier(...)
    >>> predictions = clf.predict(X_train)
    """

    def __init__(self, min_samples_leaf: int = 2,
                 confidence_factor: float = 0.25,
                 unpruned: bool = False,
                 random_state: Optional[int] = None):
        """Initialize PARTClassifier.

        Parameters
        ----------
        min_samples_leaf : int, default=2
            Minimum number of samples allowed in a leaf.
        confidence_factor : float, default=0.25
            Confidence factor used for pruning.
        unpruned : bool, default=False
            Whether pruning should be performed.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        super().__init__()
        self.min_samples_leaf = min_samples_leaf
        self.confidence_factor = confidence_factor
        self.unpruned = unpruned
        self.random_state = random_state
        self.rules_ = None
        self.classes_ = None
        self.default_class_ = None
        self._n_features = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "min_samples_leaf": {"type": "integer", "default": 2, "minimum": 1,
                                "description": "Minimum samples per leaf"},
            "confidence_factor": {"type": "number", "default": 0.25,
                                 "minimum": 0, "maximum": 1,
                                 "description": "Confidence for pruning"},
            "unpruned": {"type": "boolean", "default": False,
                        "description": "Skip pruning if True"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "missing_values", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(n^2 * m * log(n)) training, O(r * m) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        return ["Frank, E., & Witten, I.H. (1998). Generating Accurate Rule "
                "Sets Without Global Optimization. Proceedings of ICML-98."]

    def _entropy(self, y: np.ndarray) -> float:
        """Calculate the Shannon entropy of a label array.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Integer-encoded class labels.

        Returns
        -------
        entropy : float
            Shannon entropy in bits.
        """
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y.astype(int))
        probs = counts[counts > 0] / len(y)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def _gain_ratio(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Calculate the gain ratio for a binary split.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Parent node labels.
        y_left : np.ndarray
            Labels in the left child.
        y_right : np.ndarray
            Labels in the right child.

        Returns
        -------
        gain_ratio : float
            Gain ratio value for the split.
        """
        n = len(y)
        if n == 0:
            return 0.0
        n_left, n_right = len(y_left), len(y_right)

        parent_entropy = self._entropy(y)
        child_entropy = (n_left / n) * self._entropy(y_left) + \
                       (n_right / n) * self._entropy(y_right)
        ig = parent_entropy - child_entropy

        split_info = 0
        for n_i in [n_left, n_right]:
            if n_i > 0:
                p = n_i / n
                split_info -= p * np.log2(p + 1e-10)

        return ig / (split_info + 1e-10)

    def _build_partial_tree(self, X: np.ndarray, y: np.ndarray,
                            path: List[Tuple[int, str, float]]) -> Optional[PartRule]:
        """Build a partial decision tree and extract a rule from the best leaf.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for remaining instances.
        y : np.ndarray of shape (n_samples,)
            Integer-encoded class labels.
        path : list of tuple
            Current path of conditions accumulated so far.

        Returns
        -------
        rule : PartRule or None
            The extracted rule from the deepest leaf of the partial tree.
        """
        n_samples = len(y)

        if n_samples < 2 * self.min_samples_leaf or len(set(y)) == 1:
            # Make leaf
            majority = Counter(y).most_common(1)[0][0]
            correct = np.sum(y == majority)
            return PartRule(conditions=path.copy(), predicted_class=majority,
                           n_correct=correct, n_covered=n_samples)

        # Find best split
        best_gain = -np.inf
        best_feature = 0
        best_threshold = 0.0

        for feature_idx in range(self._n_features):
            X_col = X[:, feature_idx]
            valid_mask = ~np.isnan(X_col)
            if np.sum(valid_mask) < 2 * self.min_samples_leaf:
                continue

            X_valid = X_col[valid_mask]
            y_valid = y[valid_mask]
            unique_vals = np.unique(X_valid)

            if len(unique_vals) < 2:
                continue

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

            for threshold in thresholds:
                left_mask = X_valid <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue

                gain = self._gain_ratio(y_valid, y_valid[left_mask], y_valid[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        if best_gain <= 0:
            majority = Counter(y).most_common(1)[0][0]
            correct = np.sum(y == majority)
            return PartRule(conditions=path.copy(), predicted_class=majority,
                           n_correct=correct, n_covered=n_samples)

        # Split and recurse on largest branch only (partial tree)
        X_col = X[:, best_feature]
        left_mask = (X_col <= best_threshold) | np.isnan(X_col)
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        # Choose the path that leads to highest purity
        if n_left >= n_right:
            new_path = path + [(best_feature, '<=', best_threshold)]
            return self._build_partial_tree(X[left_mask], y[left_mask], new_path)
        else:
            new_path = path + [(best_feature, '>', best_threshold)]
            return self._build_partial_tree(X[right_mask], y[right_mask], new_path)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PARTClassifier":
        """Fit the PARTClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : PARTClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._n_features = X.shape[1]
        self.classes_ = np.unique(y)

        # Convert to integer indices
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y])

        # Default class
        self.default_class_ = self.classes_[Counter(y_idx).most_common(1)[0][0]]

        self.rules_ = []
        remaining_mask = np.ones(len(y), dtype=bool)

        while np.sum(remaining_mask) > 0:
            X_rem = X[remaining_mask]
            y_rem = y_idx[remaining_mask]

            if len(y_rem) < self.min_samples_leaf:
                break

            rule = self._build_partial_tree(X_rem, y_rem, [])

            if rule is None or len(rule.conditions) == 0:
                break

            # Convert prediction back to original class
            rule.predicted_class = self.classes_[rule.predicted_class]
            self.rules_.append(rule)

            # Remove covered instances
            for i in np.where(remaining_mask)[0]:
                if rule.covers(X[i]):
                    remaining_mask[i] = False

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

        predictions = np.full(len(X), self.default_class_, dtype=self.classes_.dtype)

        for i in range(len(X)):
            for rule in self.rules_:
                if rule.covers(X[i]):
                    predictions[i] = rule.predicted_class
                    break

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities (one-hot encoded hard predictions).
        """
        self._check_is_fitted()
        predictions = self.predict(X)
        n_samples = len(predictions)
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        for i, pred in enumerate(predictions):
            class_idx = np.where(self.classes_ == pred)[0]
            if len(class_idx) > 0:
                proba[i, class_idx[0]] = 1.0
        return proba

    def get_rules_description(self) -> str:
        """Get a human-readable description of the learned rules.

        Returns
        -------
        description : str
            String representation of all rules.
        """
        if not self._is_fitted:
            return "Model not fitted"
        return "\n".join([str(rule) for rule in self.rules_])

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"PARTClassifier(n_rules={len(self.rules_)})"
        return f"PARTClassifier(min_samples_leaf={self.min_samples_leaf})"
