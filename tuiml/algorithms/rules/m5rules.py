"""M5ModelRulesRegressor implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from tuiml.base.algorithms import Regressor, regressor

@dataclass
class M5ModelRule:
    """A single regression rule."""
    conditions: List[Tuple[int, str, float]] = field(default_factory=list)
    # Each condition is (attribute_index, operator, value)
    # operator: '<=' or '>'
    linear_model: Optional[np.ndarray] = None
    intercept: float = 0.0
    prediction: float = 0.0  # Constant prediction if no linear model
    coverage: int = 0  # Number of instances covered

@regressor(tags=["rules", "regression", "interpretable"], version="1.0.0")
class M5ModelRulesRegressor(Regressor):
    """M5 Model Rules for **rule-based regression** using
    **linear models** at each rule leaf.

    M5ModelRulesRegressor generates regression rules from M5 model trees.
    Each rule consists of a set of conditions and a **linear model** for
    prediction, combining the interpretability of rules with the accuracy
    of linear regression.

    Overview
    --------
    The algorithm builds regression rules via a separate-and-conquer strategy:

    1. Compute the global standard deviation of the target variable
    2. While uncovered instances remain:
       a. Find the best split using **Standard Deviation Reduction** (SDR)
       b. Recurse on the branch with lower variance, accumulating conditions
       c. When a stopping criterion is met (min samples, low variance), create
          a rule with a **linear model** fitted to the covered instances
       d. Remove covered instances and repeat
    3. Instances not covered by any rule receive the global mean prediction

    Theory
    ------
    The splitting criterion uses **Standard Deviation Reduction** (SDR):

    .. math::
        \\text{SDR}(S, A) = \\sigma(S) - \\sum_{v} \\frac{|S_v|}{|S|} \\sigma(S_v)

    where :math:`\\sigma(S)` is the standard deviation of the target in set
    :math:`S`, and :math:`S_v` are the subsets after splitting on attribute
    :math:`A`.

    At each leaf, a **ridge-regularized linear model** is fitted:

    .. math::
        \\hat{y} = X \\beta + \\beta_0, \\quad \\beta = (X^T X + \\lambda I)^{-1} X^T y

    where :math:`\\lambda` is a small regularization constant to ensure
    numerical stability.

    Parameters
    ----------
    min_samples_leaf : int, default=4
        Minimum number of samples allowed in a leaf.
    use_unsmoothed : bool, default=False
        Whether to use unsmoothed predictions.

    Attributes
    ----------
    rules_ : list of M5ModelRule
        List of learned regression rules.
    n_features_ : int
        Number of features in the training data.
    default_prediction_ : float
        Default prediction value (mean of y) when no rule matches.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n \\cdot m \\cdot \\log(n) \\cdot r)` where :math:`n`
      = number of samples, :math:`m` = number of features, and :math:`r` =
      number of rules generated
    - Prediction: :math:`O(r \\cdot m)` per sample (first matching rule)

    **When to use M5ModelRulesRegressor:**

    - Regression tasks requiring interpretable IF-THEN rules
    - When piecewise linear relationships are expected
    - Datasets with heterogeneous subregions needing different linear models
    - When a balance of accuracy and interpretability is desired

    References
    ----------
    .. [Holmes1999] Holmes, G., Hall, M. and Frank, E. (1999).
           **Generating Rule Sets from Model Trees.**
           *12th Australian Joint Conference on Artificial Intelligence*, pp. 1-12.

    .. [Quinlan1992] Quinlan, J.R. (1992).
           **Learning with Continuous Classes.**
           *Proceedings of the 5th Australian Joint Conference on Artificial Intelligence*, pp. 343-348.

    See Also
    --------
    :class:`~tuiml.algorithms.rules.RIPPERClassifier` : RIPPER rule learner for classification.
    :class:`~tuiml.algorithms.rules.PARTClassifier` : Partial-tree rule learner for classification.
    :class:`~tuiml.algorithms.rules.DecisionTableClassifier` : Decision table classifier.

    Examples
    --------
    Basic usage for rule-based regression:

    >>> from tuiml.algorithms.rules import M5ModelRulesRegressor
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X_train = np.array([[1], [2], [3], [4], [5]], dtype=float)
    >>> y_train = np.array([1.1, 2.0, 3.1, 4.0, 5.2])
    >>>
    >>> # Fit the model
    >>> reg = M5ModelRulesRegressor(min_samples_leaf=2)
    >>> reg.fit(X_train, y_train)
    M5ModelRulesRegressor(...)
    >>> predictions = reg.predict(X_train)
    >>> rules = reg.get_rules()
    """

    def __init__(
        self,
        min_samples_leaf: int = 4,
        use_unsmoothed: bool = False,
    ):
        """Initialize M5ModelRulesRegressor.

        Parameters
        ----------
        min_samples_leaf : int, default=4
            Minimum samples per leaf.
        use_unsmoothed : bool, default=False
            Use unsmoothed predictions.
        """
        super().__init__()
        self.min_samples_leaf = min_samples_leaf
        self.use_unsmoothed = use_unsmoothed

        # Fitted attributes
        self.rules_ = None
        self.n_features_ = None
        self.default_prediction_ = None
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
            "use_unsmoothed": {
                "type": "boolean",
                "default": False,
                "description": "Use unsmoothed predictions"
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
        return "O(n * m * log(n) * r) for r rules"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Holmes, G., Hall, M., & Frank, E. (1999). Generating rule sets "
            "from model trees. 12th Australian Joint Conf on AI, 1-12."
        ]

    def _compute_std(self, y: np.ndarray) -> float:
        """Compute the sample standard deviation of target values.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        std : float
            Sample standard deviation (ddof=1), or 0.0 if fewer than 2 samples.
        """
        if len(y) < 2:
            return 0.0
        return np.std(y, ddof=1)

    def _compute_sdr(self, y: np.ndarray, y_left: np.ndarray,
                     y_right: np.ndarray) -> float:
        """Compute Standard Deviation Reduction for a binary split.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Parent target values.
        y_left : np.ndarray
            Target values in the left child.
        y_right : np.ndarray
            Target values in the right child.

        Returns
        -------
        sdr : float
            Standard Deviation Reduction value.
        """
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)

        if n == 0 or n_left == 0 or n_right == 0:
            return 0.0

        std_total = self._compute_std(y)
        std_left = self._compute_std(y_left)
        std_right = self._compute_std(y_right)

        sdr = std_total - (n_left / n) * std_left - (n_right / n) * std_right
        return sdr

    def _find_best_split(self, X: np.ndarray, y: np.ndarray
                         ) -> Tuple[int, float, float, str]:
        """Find the best binary split that maximizes Standard Deviation Reduction.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        best_attr : int
            Index of the best splitting attribute (-1 if no valid split).
        best_value : float
            Threshold value for the split.
        best_sdr : float
            Standard Deviation Reduction of the best split.
        best_branch : str
            Which branch (``'<='`` or ``'>'``) has lower variance.
        """
        n_samples, n_features = X.shape
        best_sdr = -float('inf')
        best_attr = -1
        best_value = 0.0
        best_branch = '<='  # Which branch leads to the best reduction

        for attr in range(n_features):
            values = np.unique(X[:, attr])
            values = values[~np.isnan(values)]

            if len(values) < 2:
                continue

            for i in range(len(values) - 1):
                split_value = (values[i] + values[i + 1]) / 2

                left_mask = X[:, attr] <= split_value
                right_mask = ~left_mask

                y_left = y[left_mask]
                y_right = y[right_mask]

                if len(y_left) < self.min_samples_leaf or \
                   len(y_right) < self.min_samples_leaf:
                    continue

                sdr = self._compute_sdr(y, y_left, y_right)

                # Determine which branch has lower variance
                std_left = self._compute_std(y_left)
                std_right = self._compute_std(y_right)

                if sdr > best_sdr:
                    best_sdr = sdr
                    best_attr = attr
                    best_value = split_value
                    best_branch = '<=' if std_left <= std_right else '>'

        return best_attr, best_value, best_sdr, best_branch

    def _fit_linear_model(self, X: np.ndarray, y: np.ndarray
                          ) -> Tuple[np.ndarray, float]:
        """Fit a ridge-regularized linear model to the data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        coeffs : np.ndarray of shape (n_features,)
            Linear model coefficients.
        intercept : float
            Intercept term.
        """
        n_samples, n_features = X.shape

        if n_samples < n_features + 1:
            return np.zeros(n_features), np.mean(y)

        X_bias = np.column_stack([np.ones(n_samples), X])

        try:
            ridge = 1e-8
            XtX = X_bias.T @ X_bias + ridge * np.eye(n_features + 1)
            Xty = X_bias.T @ y
            coeffs = np.linalg.solve(XtX, Xty)
            intercept = coeffs[0]
            coeffs = coeffs[1:]
        except np.linalg.LinAlgError:
            return np.zeros(n_features), np.mean(y)

        return coeffs, intercept

    def _build_rule(self, X: np.ndarray, y: np.ndarray,
                    conditions: List[Tuple[int, str, float]]) -> M5ModelRule:
        """Build a single regression rule with a linear model at its leaf.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for covered instances.
        y : np.ndarray of shape (n_samples,)
            Target values for covered instances.
        conditions : list of tuple
            List of (attribute_index, operator, value) conditions.

        Returns
        -------
        rule : M5ModelRule
            A regression rule with conditions and a fitted linear model.
        """
        rule = M5ModelRule(conditions=list(conditions))

        # Fit linear model for this rule
        coeffs, intercept = self._fit_linear_model(X, y)
        rule.linear_model = coeffs
        rule.intercept = intercept
        rule.prediction = np.mean(y)
        rule.coverage = len(y)

        return rule

    def _extract_rules(self, X: np.ndarray, y: np.ndarray,
                       conditions: List[Tuple[int, str, float]],
                       rules: List[M5ModelRule]) -> np.ndarray:
        """Recursively extract regression rules from data using SDR splits.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for remaining instances.
        y : np.ndarray of shape (n_samples,)
            Target values for remaining instances.
        conditions : list of tuple
            Accumulated conditions from parent splits.
        rules : list of M5ModelRule
            List to append newly created rules to (modified in place).

        Returns
        -------
        uncovered_mask : np.ndarray of shape (n_samples,)
            Boolean mask indicating instances not covered by the new rule.
        """
        n_samples = len(y)

        # Stopping conditions
        if n_samples < 2 * self.min_samples_leaf:
            # Create leaf rule
            rule = self._build_rule(X, y, conditions)
            rules.append(rule)
            return np.zeros(n_samples, dtype=bool)

        std = self._compute_std(y)
        if std < 0.05 * self._global_std:
            # Create leaf rule
            rule = self._build_rule(X, y, conditions)
            rules.append(rule)
            return np.zeros(n_samples, dtype=bool)

        # Find best split
        best_attr, best_value, best_sdr, best_branch = self._find_best_split(X, y)

        if best_attr < 0 or best_sdr <= 0:
            # No good split, create leaf rule
            rule = self._build_rule(X, y, conditions)
            rules.append(rule)
            return np.zeros(n_samples, dtype=bool)

        # Add condition and recurse on best branch
        new_condition = (best_attr, best_branch, best_value)
        new_conditions = conditions + [new_condition]

        if best_branch == '<=':
            mask = X[:, best_attr] <= best_value
        else:
            mask = X[:, best_attr] > best_value

        X_subset = X[mask]
        y_subset = y[mask]

        # Recursively extract rules from the best branch
        self._extract_rules(X_subset, y_subset, new_conditions, rules)

        # Return the remaining (uncovered) instances
        return ~mask

    def _apply_conditions(self, x: np.ndarray,
                          conditions: List[Tuple[int, str, float]]) -> bool:
        """Check if a single sample satisfies all rule conditions.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            Single input sample.
        conditions : list of tuple
            List of (attribute_index, operator, value) conditions.

        Returns
        -------
        satisfies : bool
            True if the sample satisfies all conditions, False otherwise.
        """
        for attr, op, value in conditions:
            if np.isnan(x[attr]):
                # Handle missing values - condition fails
                return False
            if op == '<=':
                if not x[attr] <= value:
                    return False
            else:  # op == '>'
                if not x[attr] > value:
                    return False
        return True

    def fit(self, X: np.ndarray, y: np.ndarray) -> "M5ModelRulesRegressor":
        """Fit the M5ModelRulesRegressor model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : M5ModelRulesRegressor
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, self.n_features_ = X.shape
        self._global_std = self._compute_std(y)
        self.default_prediction_ = np.mean(y)

        # Extract rules iteratively (separate-and-conquer)
        self.rules_ = []
        remaining_mask = np.ones(n_samples, dtype=bool)

        while np.sum(remaining_mask) >= self.min_samples_leaf:
            X_remaining = X[remaining_mask]
            y_remaining = y[remaining_mask]

            n_rules_before = len(self.rules_)
            uncovered = self._extract_rules(X_remaining, y_remaining, [], self.rules_)

            # If no new rules were added, break
            if len(self.rules_) == n_rules_before:
                break

            # Update remaining mask based on last rule added
            last_rule = self.rules_[-1]
            for i, idx in enumerate(np.where(remaining_mask)[0]):
                if self._apply_conditions(X[idx], last_rule.conditions):
                    remaining_mask[idx] = False

            # Prevent infinite loop
            if len(self.rules_) > n_samples:
                break

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

        n_samples = X.shape[0]
        predictions = np.full(n_samples, self.default_prediction_)

        for i in range(n_samples):
            x = X[i]
            for rule in self.rules_:
                if self._apply_conditions(x, rule.conditions):
                    if rule.linear_model is not None:
                        predictions[i] = x @ rule.linear_model + rule.intercept
                    else:
                        predictions[i] = rule.prediction
                    break  # First matching rule

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the R-squared (coefficient of determination) score.

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

    def get_rules(self) -> List[str]:
        """Get a human-readable representation of the learned rules.

        Returns
        -------
        rule_strings : list of str
            List of formatted rule strings with conditions and linear models.
        """
        self._check_is_fitted()
        rule_strings = []

        for i, rule in enumerate(self.rules_):
            conditions_str = []
            for attr, op, value in rule.conditions:
                conditions_str.append(f"x[{attr}] {op} {value:.4f}")

            if conditions_str:
                condition_part = " AND ".join(conditions_str)
            else:
                condition_part = "TRUE"

            if rule.linear_model is not None:
                # Format linear model
                terms = []
                for j, coef in enumerate(rule.linear_model):
                    if abs(coef) > 1e-6:
                        terms.append(f"{coef:.4f}*x[{j}]")
                if terms:
                    model_str = " + ".join(terms) + f" + {rule.intercept:.4f}"
                else:
                    model_str = f"{rule.intercept:.4f}"
            else:
                model_str = f"{rule.prediction:.4f}"

            rule_str = f"Rule {i+1}: IF {condition_part} THEN y = {model_str} (coverage={rule.coverage})"
            rule_strings.append(rule_str)

        return rule_strings

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return f"M5ModelRulesRegressor(n_rules={len(self.rules_)})"
        return f"M5ModelRulesRegressor(min_samples_leaf={self.min_samples_leaf})"
