"""DecisionStumpClassifier classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from tuiml.base.algorithms import Classifier, classifier
from ._core import best_split_stump

@classifier(tags=["trees", "simple", "interpretable", "weak-learner"], version="1.0.0")
class DecisionStumpClassifier(Classifier):
    """Decision Stump - a one-level decision tree.

    A decision stump is a **single-split** decision tree that selects the
    best attribute and split point to minimize **classification error**. Decision
    stumps are commonly used as **weak learners** in ensemble methods such as
    AdaBoost and bagging.

    Overview
    --------
    The algorithm builds a depth-1 tree:

    1. For each feature, evaluate all candidate split points
    2. For numeric features, find the threshold that **minimizes weighted
       misclassification error**
    3. For nominal features, group attribute values by majority class
    4. Select the single best feature and split across all candidates
    5. Assign the **majority class** to each branch

    Theory
    ------
    For a numeric attribute with threshold :math:`t`, the weighted error is:

    .. math::
        E(t) = \\frac{1}{W} \\left( \\sum_{x_j \\leq t} w_j \\cdot \\mathbb{1}[y_j \\neq \\hat{y}_L]
        + \\sum_{x_j > t} w_j \\cdot \\mathbb{1}[y_j \\neq \\hat{y}_R] \\right)

    where :math:`W = \\sum w_j` is the total weight, :math:`\\hat{y}_L` and
    :math:`\\hat{y}_R` are the majority classes of the left and right branches,
    and :math:`\\mathbb{1}[\\cdot]` is the indicator function.

    Parameters
    ----------
    (No user-configurable parameters.)

    Attributes
    ----------
    feature_index_ : int
        Index of the feature used for splitting.
    threshold_ : float
        Threshold value for numeric features.
    is_numeric_ : bool
        Whether the split feature is numeric.
    left_class_ : Any
        Class predicted when condition is True (value <= threshold).
    right_class_ : Any
        Class predicted when condition is False (value > threshold).
    left_distribution_ : dict
        Class distribution for the left branch.
    right_distribution_ : dict
        Class distribution for the right branch.
    classes_ : np.ndarray
        Unique class labels.
    n_samples_ : int
        Number of training samples seen during ``fit()``.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot m \\cdot \\log(n))` where :math:`n` = samples,
      :math:`m` = features (sorting per feature)
    - Prediction: :math:`O(1)` per sample (single comparison)

    **When to use DecisionStumpClassifier:**

    - As a **weak learner** in boosting ensembles (e.g., AdaBoost)
    - When you need the simplest possible interpretable model
    - As a baseline classifier for benchmarking
    - When training speed is critical and high accuracy is not required

    References
    ----------
    .. [Iba1992] Iba, W. and Langley, P. (1992).
           **Induction of One-Level Decision Trees.**
           *Proceedings of the 9th International Conference on Machine Learning*,
           pp. 233-240.

    .. [Freund1997] Freund, Y. and Schapire, R.E. (1997).
           **A Decision-Theoretic Generalization of On-Line Learning and an
           Application to Boosting.**
           *Journal of Computer and System Sciences*, 55(1), pp. 119-139.
           DOI: `10.1006/jcss.1997.1504 <https://doi.org/10.1006/jcss.1997.1504>`_

    See Also
    --------
    :class:`~tuiml.algorithms.trees.C45TreeClassifier` : Full C4.5 decision tree with pruning.
    :class:`~tuiml.algorithms.trees.RandomForestClassifier` : Ensemble of randomized trees.
    :class:`~tuiml.algorithms.trees.HoeffdingTreeClassifier` : Incremental tree for streaming data.

    Examples
    --------
    Basic usage as a standalone classifier:

    >>> from tuiml.algorithms.trees import DecisionStumpClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 0, 1, 1])
    >>>
    >>> # Fit the stump
    >>> clf = DecisionStumpClassifier()
    >>> clf.fit(X, y)
    DecisionStumpClassifier(...)
    >>> predictions = clf.predict(X)
    """

    def __init__(self):
        """Initialize DecisionStumpClassifier classifier."""
        super().__init__()
        self.feature_index_ = None
        self.threshold_ = None
        self.is_numeric_ = None
        self.left_class_ = None
        self.right_class_ = None
        self.left_distribution_ = None
        self.right_distribution_ = None
        self.classes_ = None
        self.n_samples_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema (DecisionStumpClassifier has no parameters)."""
        return {}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return classifier capabilities."""
        return [
            "numeric",
            "nominal",
            "binary_class",
            "multiclass"
        ]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * m * log(n)) training, O(1) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Iba, W., & Langley, P. (1992). Induction of One-Level Decision Trees. "
            "Proceedings of the 9th International Conference on Machine Learning."
        ]

    def _is_numeric_attribute(self, values: np.ndarray) -> bool:
        """Check if an attribute is numeric.

        Parameters
        ----------
        values : np.ndarray
            Attribute values to check.

        Returns
        -------
        is_numeric : bool
            True if values can be cast to float and are not all NaN.
        """
        try:
            values = np.asarray(values, dtype=float)
            return not np.isnan(values).all()
        except (ValueError, TypeError):
            return False

    def _compute_class_distribution(self, y: np.ndarray) -> Dict:
        """Compute class distribution from labels.

        Parameters
        ----------
        y : np.ndarray
            Array of class labels.

        Returns
        -------
        distribution : dict
            Mapping from class label to proportion.
        """
        distribution = {}
        total = len(y)
        if total == 0:
            return {c: 0.0 for c in self.classes_}

        for cls in self.classes_:
            distribution[cls] = np.sum(y == cls) / total
        return distribution

    def _find_best_nominal_split(self, X_attr: np.ndarray, y: np.ndarray,
                                  sample_weight: Optional[np.ndarray] = None
                                  ) -> Tuple[set, float]:
        """Find the best value subset for a nominal attribute.

        Parameters
        ----------
        X_attr : np.ndarray of shape (n_samples,)
            Single attribute values.
        y : np.ndarray of shape (n_samples,)
            Class labels.
        sample_weight : np.ndarray of shape (n_samples,) or None, default=None
            Sample weights.

        Returns
        -------
        left_values : set
            Set of attribute values assigned to the left branch.
        error : float
            Weighted classification error.
        """
        if sample_weight is None:
            sample_weight = np.ones(len(y))

        unique_values = np.unique(X_attr)

        # For each value, compute class distribution
        value_class_counts = {}
        for val in unique_values:
            mask = X_attr == val
            value_class_counts[val] = defaultdict(float)
            for i, m in enumerate(mask):
                if m:
                    value_class_counts[val][y[i]] += sample_weight[i]

        # Find majority class for each value
        value_majority = {}
        for val, counts in value_class_counts.items():
            if counts:
                value_majority[val] = max(counts, key=counts.get)
            else:
                value_majority[val] = self.classes_[0]

        # Group values by their majority class
        class_groups = defaultdict(set)
        for val, maj_class in value_majority.items():
            class_groups[maj_class].add(val)

        if class_groups:
            left_class = max(class_groups, key=lambda c: len(class_groups[c]))
            left_values = class_groups[left_class]
        else:
            left_values = set()

        # Calculate error
        total_weight = sum(sample_weight)
        error = 0
        for val in unique_values:
            mask = X_attr == val
            for i, m in enumerate(mask):
                if m:
                    predicted = value_majority[val]
                    if y[i] != predicted:
                        error += sample_weight[i]

        error /= total_weight if total_weight > 0 else 1

        return left_values, error

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> "DecisionStumpClassifier":
        """Fit the DecisionStumpClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.
        sample_weight : np.ndarray of shape (n_samples,), optional
            Sample weights (for boosting applications).

        Returns
        -------
        self : DecisionStumpClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(n_samples)

        # Check which features are numeric
        is_numeric = [self._is_numeric_attribute(X[:, i]) for i in range(n_features)]

        # Try numeric features via _core.best_split_stump
        numeric_mask = np.array(is_numeric)
        best_error = float('inf')
        best_feature = 0
        best_threshold = None
        best_is_numeric = True
        best_left_values = None

        if np.any(numeric_mask):
            # Use core stump splitter for numeric features
            numeric_indices = np.where(numeric_mask)[0]
            X_numeric = X[:, numeric_indices].astype(float)
            feat_idx, threshold, error, _ = best_split_stump(
                X_numeric, y, sample_weight
            )
            if error < best_error:
                best_error = error
                best_feature = numeric_indices[feat_idx]
                best_threshold = threshold
                best_is_numeric = True
                best_left_values = None

        # Try nominal features
        for feature_idx in range(n_features):
            if is_numeric[feature_idx]:
                continue
            X_attr = X[:, feature_idx]
            left_values, error = self._find_best_nominal_split(
                X_attr, y, sample_weight
            )
            if error < best_error:
                best_error = error
                best_feature = feature_idx
                best_threshold = None
                best_is_numeric = False
                best_left_values = left_values

        self.feature_index_ = best_feature
        self.threshold_ = best_threshold
        self.is_numeric_ = best_is_numeric
        self._left_values = best_left_values

        # Compute class predictions for each branch
        X_attr = X[:, self.feature_index_]
        if self.is_numeric_:
            X_attr_float = X_attr.astype(float)
            left_mask = X_attr_float <= self.threshold_
        else:
            left_mask = np.isin(X_attr, list(self._left_values) if self._left_values else [])

        right_mask = ~left_mask

        # Compute distributions and majority classes
        left_y = y[left_mask]
        right_y = y[right_mask]
        left_weights = sample_weight[left_mask]
        right_weights = sample_weight[right_mask]

        # Left branch
        if len(left_y) > 0:
            left_class_counts = defaultdict(float)
            for i, cls in enumerate(left_y):
                left_class_counts[cls] += left_weights[i]
            self.left_class_ = max(left_class_counts, key=left_class_counts.get)
            total_left = sum(left_class_counts.values())
            self.left_distribution_ = {c: left_class_counts[c] / total_left
                                       for c in self.classes_}
        else:
            self.left_class_ = self.classes_[0]
            self.left_distribution_ = {c: 1.0 / len(self.classes_) for c in self.classes_}

        # Right branch
        if len(right_y) > 0:
            right_class_counts = defaultdict(float)
            for i, cls in enumerate(right_y):
                right_class_counts[cls] += right_weights[i]
            self.right_class_ = max(right_class_counts, key=right_class_counts.get)
            total_right = sum(right_class_counts.values())
            self.right_distribution_ = {c: right_class_counts[c] / total_right
                                        for c in self.classes_}
        else:
            self.right_class_ = self.classes_[0]
            self.right_distribution_ = {c: 1.0 / len(self.classes_) for c in self.classes_}

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes using the decision stump.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted classes.
        """
        self._check_is_fitted()
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=self.classes_.dtype)

        X_attr = X[:, self.feature_index_]

        if self.is_numeric_:
            X_attr_float = X_attr.astype(float)
            left_mask = X_attr_float <= self.threshold_
        else:
            left_mask = np.isin(X_attr, list(self._left_values) if self._left_values else [])

        predictions[left_mask] = self.left_class_
        predictions[~left_mask] = self.right_class_

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
            Class probabilities.
        """
        self._check_is_fitted()
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))

        X_attr = X[:, self.feature_index_]

        if self.is_numeric_:
            X_attr_float = X_attr.astype(float)
            left_mask = X_attr_float <= self.threshold_
        else:
            left_mask = np.isin(X_attr, list(self._left_values) if self._left_values else [])

        for i, cls in enumerate(self.classes_):
            proba[left_mask, i] = self.left_distribution_.get(cls, 0)
            proba[~left_mask, i] = self.right_distribution_.get(cls, 0)

        return proba

    def get_stump_description(self) -> str:
        """Get a human-readable description of the decision stump.

        Returns
        -------
        description : str
            Human-readable description of the stump rule.
        """
        if not self._is_fitted:
            return "Model not fitted"

        lines = ["Decision Stump:"]

        if self.is_numeric_:
            lines.append(f"  If feature[{self.feature_index_}] <= {self.threshold_:.4f}:")
            lines.append(f"    Predict: {self.left_class_}")
            lines.append(f"  Else:")
            lines.append(f"    Predict: {self.right_class_}")
        else:
            left_vals = list(self._left_values) if self._left_values else []
            lines.append(f"  If feature[{self.feature_index_}] in {left_vals}:")
            lines.append(f"    Predict: {self.left_class_}")
            lines.append(f"  Else:")
            lines.append(f"    Predict: {self.right_class_}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            if self.is_numeric_:
                return f"DecisionStumpClassifier(feature={self.feature_index_}, threshold={self.threshold_:.4f})"
            else:
                return f"DecisionStumpClassifier(feature={self.feature_index_}, nominal=True)"
        return "DecisionStumpClassifier()"
