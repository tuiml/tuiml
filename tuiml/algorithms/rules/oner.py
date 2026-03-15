"""OneRuleClassifier (One Rule) classifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from tuiml.base.algorithms import Classifier, classifier

@classifier(tags=["rules", "simple", "interpretable"], version="1.0.0")
class OneRuleClassifier(Classifier):
    """One Rule classifier using the **single best attribute** for
    **minimum-error classification**.

    OneRuleClassifier is a simple, accurate classification algorithm that
    generates one rule for each attribute in the training data and selects
    the rule with the smallest error rate. For numeric attributes, it uses
    **supervised discretization** to create rules.

    Overview
    --------
    The algorithm selects the single most predictive attribute as follows:

    1. For each attribute in the dataset:
       a. If numeric, discretize into buckets of at least ``min_bucket_size``
          instances using class-boundary splits
       b. For each unique value (or bucket), assign the majority class
       c. Compute the total error rate for this attribute
    2. Select the attribute with the lowest error rate
    3. At prediction time, look up the value of the selected attribute and
       return the associated majority class

    Theory
    ------
    For each attribute :math:`a`, the error rate is computed as:

    .. math::
        \\text{Error}(a) = 1 - \\frac{1}{n} \\sum_{v \\in \\text{vals}(a)} \\max_{c \\in C} n_{v,c}

    where :math:`n_{v,c}` is the number of instances with attribute value
    :math:`v` belonging to class :math:`c`, and the best attribute is:

    .. math::
        a^* = \\arg\\min_{a \\in A} \\text{Error}(a)

    For numeric attributes, supervised discretization creates breakpoints
    where the majority class changes, subject to a minimum bucket size
    constraint.

    Parameters
    ----------
    min_bucket_size : int, default=6
        Minimum number of instances per bucket for numeric attribute
        discretization.

    Attributes
    ----------
    best_attribute_ : int
        Index of the best attribute.
    rules_ : dict
        Dictionary mapping attribute values to predicted classes.
    attribute_error_ : float
        Error rate of the best attribute.
    is_numeric_ : np.ndarray
        Boolean array indicating numeric attributes.
    breakpoints_ : list of float or None
        Breakpoints used for discretizing the best numeric attribute.
    classes_ : np.ndarray
        Unique class labels encountered during training.
    default_class_ : any
        Default class prediction (majority class).

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot m \\cdot \\log(n))` where :math:`n` = number
      of samples and :math:`m` = number of features (sorting for
      discretization dominates)
    - Prediction: :math:`O(m)` per sample (single attribute lookup)

    **When to use OneRuleClassifier:**

    - As a simple interpretable baseline
    - When one attribute is expected to be highly predictive
    - For quick feature importance assessment
    - When the simplest possible model is desired for explanation

    References
    ----------
    .. [Holte1993] Holte, R.C. (1993).
           **Very Simple Classification Rules Perform Well on Most Commonly Used Datasets.**
           *Machine Learning*, 11, pp. 63-91.
           DOI: `10.1023/A:1022631118932 <https://doi.org/10.1023/A:1022631118932>`_

    See Also
    --------
    :class:`~tuiml.algorithms.rules.ZeroRuleClassifier` : Majority-class baseline (no attributes used).
    :class:`~tuiml.algorithms.rules.DecisionTableClassifier` : Multi-attribute lookup table classifier.
    :class:`~tuiml.algorithms.rules.RIPPERClassifier` : Multi-rule propositional learner.

    Examples
    --------
    Basic usage for single-attribute classification:

    >>> from tuiml.algorithms.rules import OneRuleClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    >>> y_train = np.array([0, 0, 1, 1, 1])
    >>>
    >>> # Fit the model
    >>> clf = OneRuleClassifier(min_bucket_size=6)
    >>> clf.fit(X_train, y_train)
    OneRuleClassifier(...)
    >>> predictions = clf.predict(X_train)
    """

    def __init__(self, min_bucket_size: int = 6):
        """Initialize OneRuleClassifier classifier.

        Parameters
        ----------
        min_bucket_size : int, default=6
            Minimum instances per bucket for numeric discretization.
        """
        super().__init__()
        self.min_bucket_size = min_bucket_size
        self.best_attribute_ = None
        self.rules_ = None
        self.attribute_error_ = None
        self.is_numeric_ = None
        self.breakpoints_ = None
        self.classes_ = None
        self.default_class_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "min_bucket_size": {
                "type": "integer",
                "default": 6,
                "minimum": 1,
                "description": "Minimum number of instances per bucket for "
                              "numeric attribute discretization"
            }
        }

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
        return "O(n * m * log(n)) training, O(m) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Holte, R.C. (1993). Very Simple Classification Rules Perform "
            "Well on Most Commonly Used Datasets. Machine Learning, 11, 63-91."
        ]

    def _is_numeric_attribute(self, values: np.ndarray) -> bool:
        """Check if an attribute column contains numeric values.

        Parameters
        ----------
        values : np.ndarray
            Column of attribute values.

        Returns
        -------
        is_numeric : bool
            True if the attribute is numeric, False otherwise.
        """
        try:
            values = np.asarray(values, dtype=float)
            return not np.isnan(values).all()
        except (ValueError, TypeError):
            return False

    def _discretize_numeric(self, values: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Discretize a numeric attribute into buckets using class-boundary splits.

        Uses minimum bucket size constraints to create breakpoints where
        the majority class changes.

        Parameters
        ----------
        values : np.ndarray of shape (n_samples,)
            Numeric attribute values.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        discretized : np.ndarray of shape (n_samples,)
            Integer bucket indices for each instance.
        breakpoints : list of float
            Breakpoint values used for discretization.
        """
        # Sort by attribute value
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_y = y[sorted_indices]

        breakpoints = []
        current_bucket_size = 0
        current_bucket_class_counts = defaultdict(int)

        for i in range(len(sorted_values)):
            current_bucket_size += 1
            current_bucket_class_counts[sorted_y[i]] += 1

            # Check if we should create a breakpoint
            if current_bucket_size >= self.min_bucket_size:
                # Look for class changes
                if i < len(sorted_values) - 1:
                    next_class = sorted_y[i + 1]
                    current_majority = max(current_bucket_class_counts,
                                          key=current_bucket_class_counts.get)

                    if next_class != current_majority or (i + 1) - (len(breakpoints) * self.min_bucket_size if breakpoints else 0) >= self.min_bucket_size * 2:
                        # Create breakpoint between current and next value
                        breakpoint = (sorted_values[i] + sorted_values[i + 1]) / 2
                        breakpoints.append(breakpoint)
                        current_bucket_size = 0
                        current_bucket_class_counts = defaultdict(int)

        # Convert values to bucket indices
        discretized = np.digitize(values, breakpoints)
        return discretized, breakpoints

    def _create_rules_for_attribute(self, attr_values: np.ndarray, y: np.ndarray,
                                    is_numeric: bool) -> Tuple[Dict, float, Optional[List[float]]]:
        """Create classification rules for a single attribute.

        Parameters
        ----------
        attr_values : np.ndarray of shape (n_samples,)
            Values of the attribute.
        y : np.ndarray of shape (n_samples,)
            Target labels.
        is_numeric : bool
            Whether the attribute is numeric.

        Returns
        -------
        rules : dict
            Dictionary mapping attribute values to predicted classes.
        error_rate : float
            Error rate of the rules for this attribute.
        breakpoints : list of float or None
            Breakpoints for numeric attributes, None for nominal.
        """
        if is_numeric:
            discretized, breakpoints = self._discretize_numeric(attr_values.astype(float), y)
            values = discretized
        else:
            values = attr_values
            breakpoints = None

        # Count class frequencies for each value
        value_class_counts = defaultdict(lambda: defaultdict(int))
        for val, cls in zip(values, y):
            value_class_counts[val][cls] += 1

        # Create rules - predict majority class for each value
        rules = {}
        correct = 0
        total = len(y)

        for val, class_counts in value_class_counts.items():
            majority_class = max(class_counts, key=class_counts.get)
            rules[val] = majority_class
            correct += class_counts[majority_class]

        error_rate = 1 - (correct / total) if total > 0 else 1.0
        return rules, error_rate, breakpoints

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OneRuleClassifier":
        """Fit the OneRuleClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : OneRuleClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)

        # Find default class (majority)
        unique, counts = np.unique(y, return_counts=True)
        self.default_class_ = unique[np.argmax(counts)]

        # Determine which attributes are numeric
        self.is_numeric_ = np.array([self._is_numeric_attribute(X[:, i])
                                     for i in range(n_features)])

        # Evaluate each attribute
        best_error = float('inf')
        best_attr = 0
        best_rules = {}
        best_breakpoints = None

        for attr_idx in range(n_features):
            attr_values = X[:, attr_idx]
            is_numeric = self.is_numeric_[attr_idx]

            rules, error_rate, breakpoints = self._create_rules_for_attribute(
                attr_values, y, is_numeric
            )

            if error_rate < best_error:
                best_error = error_rate
                best_attr = attr_idx
                best_rules = rules
                best_breakpoints = breakpoints

        self.best_attribute_ = best_attr
        self.rules_ = best_rules
        self.attribute_error_ = best_error
        self.breakpoints_ = best_breakpoints

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes using the one rule.

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
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]

        # Get values for the best attribute
        attr_values = X[:, self.best_attribute_]

        # Discretize if numeric
        if self.is_numeric_[self.best_attribute_] and self.breakpoints_ is not None:
            attr_values = np.digitize(attr_values.astype(float), self.breakpoints_)

        # Apply rules
        predictions = [self.rules_.get(val, self.default_class_) for val in attr_values]
        return np.array(predictions, dtype=self.classes_.dtype)

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

    def get_rule_description(self) -> str:
        """Get a human-readable description of the learned rule.

        Returns
        -------
        description : str
            String representation of the one-rule classifier.
        """
        if not self._is_fitted:
            return "Model not fitted"

        lines = [f"OneRuleClassifier rule for attribute {self.best_attribute_}:"]

        if self.is_numeric_[self.best_attribute_] and self.breakpoints_:
            # Describe ranges
            breakpoints = [-np.inf] + list(self.breakpoints_) + [np.inf]
            for i in range(len(breakpoints) - 1):
                bucket_id = i
                if bucket_id in self.rules_:
                    low = breakpoints[i]
                    high = breakpoints[i + 1]
                    if low == -np.inf:
                        condition = f"  <= {high:.4f}"
                    elif high == np.inf:
                        condition = f"  > {breakpoints[i]:.4f}"
                    else:
                        condition = f"  ({low:.4f}, {high:.4f}]"
                    lines.append(f"{condition} -> {self.rules_[bucket_id]}")
        else:
            for val, pred in sorted(self.rules_.items(), key=lambda x: str(x[0])):
                lines.append(f"  = {val} -> {pred}")

        lines.append(f"\nError rate: {self.attribute_error_:.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"OneRuleClassifier(best_attribute={self.best_attribute_}, "
                   f"error_rate={self.attribute_error_:.4f})")
        return f"OneRuleClassifier(min_bucket_size={self.min_bucket_size})"
