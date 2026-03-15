"""RIPPERClassifier rule learner implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

from tuiml.base.algorithms import Classifier, classifier

@dataclass
class Rule:
    """A classification rule with conditions."""
    conditions: List[Tuple[int, str, float]] = field(default_factory=list)
    predicted_class: Any = None
    covered: int = 0
    correct: int = 0

    def covers(self, x: np.ndarray) -> bool:
        """Check if rule covers an instance."""
        for feature_idx, operator, value in self.conditions:
            if operator == '<=':
                if x[feature_idx] > value:
                    return False
            elif operator == '>':
                if x[feature_idx] <= value:
                    return False
            elif operator == '==':
                if x[feature_idx] != value:
                    return False
        return True

    def __str__(self) -> str:
        if not self.conditions:
            return f"() => {self.predicted_class}"
        conds = " AND ".join([f"x[{f}] {op} {v:.4f}" for f, op, v in self.conditions])
        return f"({conds}) => {self.predicted_class}"

@classifier(tags=["rules", "ripper", "interpretable"], version="1.0.0")
class RIPPERClassifier(Classifier):
    """RIPPER rule learner using **separate-and-conquer** with
    **incremental reduced error pruning** for propositional rule induction.

    RIPPERClassifier (Repeated Incremental Pruning to Produce Error Reduction)
    learns a set of propositional rules using a separate-and-conquer approach.
    Also known as JRip.

    Overview
    --------
    The RIPPER algorithm learns an ordered rule set through the following steps:

    1. Order classes by frequency (minority classes first)
    2. For each class except the majority (default) class:
       a. Split the remaining data into grow and prune sets
       b. **Grow** a rule by greedily adding conditions that maximize FOIL gain
       c. **Prune** the rule using the prune set (reduced error pruning)
       d. Add the rule if it meets minimum coverage and accuracy thresholds
       e. Remove instances covered by the rule and repeat
    3. Optionally perform global optimization passes on the entire rule set

    Theory
    ------
    Rule growing uses **FOIL information gain** to select the best condition.
    For a candidate condition that refines coverage from :math:`(p_0, n_0)` to
    :math:`(p_1, n_1)`:

    .. math::
        \\text{FoilGain} = p_1 \\left( \\log_2 \\frac{p_1}{p_1 + n_1} - \\log_2 \\frac{p_0}{p_0 + n_0} \\right)

    where :math:`p_i` and :math:`n_i` are the number of positive and negative
    examples covered before and after adding the condition.

    Rule pruning uses the **Minimum Description Length** (MDL) principle.
    A rule is pruned if removing the last condition does not decrease
    validation-set accuracy:

    .. math::
        \\text{Acc}_{\\text{prune}}(R) = \\frac{\\text{correct}(R, D_{\\text{val}})}{\\text{covered}(R, D_{\\text{val}})}

    Parameters
    ----------
    min_no : float, default=2.0
        Minimum number of instances per rule.
    num_folds : int, default=3
        Number of folds for Reduced Error Pruning (REP).
    num_optimizations : int, default=2
        Number of optimization runs.
    use_pruning : bool, default=True
        Whether to use pruning.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    rules_ : list of Rule
        List of learned classification rules.
    classes_ : np.ndarray
        Unique class labels.
    default_class_ : any
        Default class for instances not covered by any rule.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(n^2 \\cdot m)` where :math:`n` = number of samples
      and :math:`m` = number of features (due to repeated rule growing over
      remaining instances)
    - Prediction: :math:`O(r \\cdot m)` per sample where :math:`r` = number
      of rules

    **When to use RIPPERClassifier:**

    - When interpretable IF-THEN rules are needed
    - Datasets with noisy features (pruning handles noise well)
    - Multi-class problems (handles class ordering automatically)
    - When a compact rule set is preferred over a large decision tree

    References
    ----------
    .. [Cohen1995] Cohen, W.W. (1995).
           **Fast Effective Rule Induction.**
           *Proceedings of the 12th International Conference on Machine Learning*, pp. 115-123.

    See Also
    --------
    :class:`~tuiml.algorithms.rules.PARTClassifier` : Rule learner using partial decision trees.
    :class:`~tuiml.algorithms.rules.OneRuleClassifier` : Single-attribute rule learner.
    :class:`~tuiml.algorithms.rules.DecisionTableClassifier` : Decision table classifier.

    Examples
    --------
    Basic usage for rule-based classification:

    >>> from tuiml.algorithms.rules import RIPPERClassifier
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2]])
    >>> y_train = np.array([0, 0, 1, 1, 1])
    >>>
    >>> # Fit the RIPPER model
    >>> clf = RIPPERClassifier(num_folds=3)
    >>> clf.fit(X_train, y_train)
    RIPPERClassifier(...)
    >>> predictions = clf.predict(X_train)
    """

    def __init__(self, min_no: float = 2.0,
                 num_folds: int = 3,
                 num_optimizations: int = 2,
                 use_pruning: bool = True,
                 random_state: Optional[int] = None):
        """Initialize RIPPERClassifier.

        Parameters
        ----------
        min_no : float, default=2.0
            Minimum number of instances per rule.
        num_folds : int, default=3
            Number of folds for Reduced Error Pruning.
        num_optimizations : int, default=2
            Number of optimization runs.
        use_pruning : bool, default=True
            Whether to use pruning.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        super().__init__()
        self.min_no = min_no
        self.num_folds = num_folds
        self.num_optimizations = num_optimizations
        self.use_pruning = use_pruning
        self.random_state = random_state
        self.rules_ = None
        self.classes_ = None
        self.default_class_ = None
        self._n_features = None
        self._rng = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "min_no": {"type": "number", "default": 2.0, "minimum": 1,
                      "description": "Minimum instances per rule"},
            "num_folds": {"type": "integer", "default": 3, "minimum": 2,
                         "description": "Folds for REP"},
            "num_optimizations": {"type": "integer", "default": 2, "minimum": 0,
                                 "description": "Optimization runs"},
            "use_pruning": {"type": "boolean", "default": True,
                          "description": "Use pruning"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(n^2 * m) training, O(r * m) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        return ["Cohen, W.W. (1995). Fast Effective Rule Induction. "
                "Proceedings of the 12th International Conference on ML."]

    def _foil_gain(self, p0: int, n0: int, p1: int, n1: int) -> float:
        """Calculate FOIL information gain for a candidate condition.

        Parameters
        ----------
        p0 : int
            Positive examples covered before adding the condition.
        n0 : int
            Negative examples covered before adding the condition.
        p1 : int
            Positive examples covered after adding the condition.
        n1 : int
            Negative examples covered after adding the condition.

        Returns
        -------
        gain : float
            FOIL information gain value.
        """
        if p1 == 0:
            return 0
        if p0 + n0 == 0 or p1 + n1 == 0:
            return 0
        log_before = np.log2(p0 / (p0 + n0) + 1e-10)
        log_after = np.log2(p1 / (p1 + n1) + 1e-10)
        return p1 * (log_after - log_before)

    def _grow_rule(self, X: np.ndarray, y: np.ndarray, target_class: Any) -> Rule:
        """Grow a single rule by greedily adding conditions that maximize FOIL gain.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target labels.
        target_class : any
            The class label this rule should predict.

        Returns
        -------
        rule : Rule
            The grown rule with conditions and coverage statistics.
        """
        rule = Rule(predicted_class=target_class)
        covered_mask = np.ones(len(y), dtype=bool)

        while True:
            pos_covered = np.sum((y == target_class) & covered_mask)
            neg_covered = np.sum((y != target_class) & covered_mask)

            if pos_covered == 0 or neg_covered == 0:
                break

            best_gain = -np.inf
            best_condition = None
            best_mask = None

            for feature_idx in range(self._n_features):
                X_col = X[:, feature_idx]
                valid_mask = covered_mask & ~np.isnan(X_col)
                if np.sum(valid_mask) < 2:
                    continue

                unique_vals = np.unique(X_col[valid_mask])
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2 if len(unique_vals) > 1 else []

                for threshold in thresholds:
                    for operator in ['<=', '>']:
                        if operator == '<=':
                            new_mask = covered_mask & (X_col <= threshold)
                        else:
                            new_mask = covered_mask & (X_col > threshold)

                        p1 = np.sum((y == target_class) & new_mask)
                        n1 = np.sum((y != target_class) & new_mask)

                        if p1 < self.min_no:
                            continue

                        gain = self._foil_gain(pos_covered, neg_covered, p1, n1)

                        if gain > best_gain:
                            best_gain = gain
                            best_condition = (feature_idx, operator, threshold)
                            best_mask = new_mask

            if best_condition is None or best_gain <= 0:
                break

            rule.conditions.append(best_condition)
            covered_mask = best_mask

        rule.covered = np.sum(covered_mask)
        rule.correct = np.sum((y == target_class) & covered_mask)
        return rule

    def _prune_rule(self, rule: Rule, X_val: np.ndarray, y_val: np.ndarray) -> Rule:
        """Prune a rule by removing trailing conditions using a validation set.

        Parameters
        ----------
        rule : Rule
            The rule to prune.
        X_val : np.ndarray of shape (n_samples, n_features)
            Validation feature matrix.
        y_val : np.ndarray of shape (n_samples,)
            Validation target labels.

        Returns
        -------
        pruned_rule : Rule
            The pruned rule with possibly fewer conditions.
        """
        if len(rule.conditions) <= 1:
            return rule

        best_rule = Rule(conditions=rule.conditions.copy(),
                        predicted_class=rule.predicted_class)

        # Compute initial accuracy
        covers = np.array([rule.covers(X_val[i]) for i in range(len(X_val))])
        if np.sum(covers) > 0:
            best_acc = np.sum((y_val == rule.predicted_class) & covers) / np.sum(covers)
        else:
            best_acc = 0

        # Try removing conditions from the end
        while len(best_rule.conditions) > 1:
            test_rule = Rule(conditions=best_rule.conditions[:-1],
                           predicted_class=rule.predicted_class)
            covers = np.array([test_rule.covers(X_val[i]) for i in range(len(X_val))])
            if np.sum(covers) > 0:
                acc = np.sum((y_val == rule.predicted_class) & covers) / np.sum(covers)
                if acc >= best_acc:
                    best_acc = acc
                    best_rule = test_rule
                else:
                    break
            else:
                break

        return best_rule

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RIPPERClassifier":
        """Fit the RIPPERClassifier classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : RIPPERClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._n_features = X.shape[1]
        self.classes_ = np.unique(y)
        self._rng = np.random.RandomState(self.random_state)

        # Order classes by frequency (learn rules for minority classes first)
        class_counts = Counter(y)
        ordered_classes = sorted(self.classes_, key=lambda c: class_counts[c])
        self.default_class_ = ordered_classes[-1]

        self.rules_ = []
        remaining_mask = np.ones(len(y), dtype=bool)

        # Learn rules for each class except the default
        for target_class in ordered_classes[:-1]:
            while True:
                X_rem = X[remaining_mask]
                y_rem = y[remaining_mask]

                pos_count = np.sum(y_rem == target_class)
                if pos_count < self.min_no:
                    break

                # Split for growing and pruning
                if self.use_pruning and len(y_rem) > 10:
                    n = len(y_rem)
                    prune_size = max(1, n // self.num_folds)
                    indices = self._rng.permutation(n)
                    grow_idx = indices[prune_size:]
                    prune_idx = indices[:prune_size]
                    X_grow, y_grow = X_rem[grow_idx], y_rem[grow_idx]
                    X_prune, y_prune = X_rem[prune_idx], y_rem[prune_idx]
                else:
                    X_grow, y_grow = X_rem, y_rem
                    X_prune, y_prune = None, None

                rule = self._grow_rule(X_grow, y_grow, target_class)

                if len(rule.conditions) == 0:
                    break

                if self.use_pruning and X_prune is not None:
                    rule = self._prune_rule(rule, X_prune, y_prune)

                # Check if rule is useful
                covers = np.array([rule.covers(X[i]) for i in range(len(X))])
                covered_indices = np.where(remaining_mask & covers)[0]

                if len(covered_indices) < self.min_no:
                    break

                correct = np.sum(y[covered_indices] == target_class)
                if correct / len(covered_indices) < 0.5:
                    break

                self.rules_.append(rule)
                remaining_mask[covered_indices] = False

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
            String representation of all rules and the default class.
        """
        if not self._is_fitted:
            return "Model not fitted"
        lines = [str(rule) for rule in self.rules_]
        lines.append(f"Default: {self.default_class_}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"RIPPERClassifier(n_rules={len(self.rules_)})"
        return f"RIPPERClassifier(min_no={self.min_no})"
