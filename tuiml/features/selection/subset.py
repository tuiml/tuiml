"""
Subset-based feature selection methods.

This module provides feature selectors that evaluate subsets of features
rather than individual features.
- CFSSelector: CfsSubsetEval.java
- WrapperSelector: WrapperSubsetEval.java
"""

import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import deque

from tuiml.base.features import FeatureSelector, feature_selector
from ._base import SelectorMixin, _ensure_numpy
from tuiml.evaluation.metrics.information_theoretic import (
    entropy as _entropy,
    conditional_entropy as _conditional_entropy,
)

def _discretize_continuous(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Discretize continuous values into equal-frequency bins.

    Parameters
    ----------
    x : ndarray
        Input continuous values.
    n_bins : int, default=10
        Number of bins for discretization.

    Returns
    -------
    result : ndarray of int
        Discretized bin indices.
    """
    x = np.asarray(x)
    nan_mask = np.isnan(x)
    if np.all(nan_mask):
        return np.zeros_like(x, dtype=int)
    x_valid = x[~nan_mask]
    n_unique = len(np.unique(x_valid))
    if n_unique <= n_bins:
        result = x.copy()
        result[nan_mask] = -999
        return result.astype(int)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(x_valid, percentiles)
    result = np.digitize(x, bin_edges[:-1])
    result = np.where(nan_mask, -999, result)
    return result.astype(int)

def _symmetrical_uncertainty(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute symmetrical uncertainty between two variables.

    SU(X, Y) = 2 * I(X; Y) / (H(X) + H(Y))

    Parameters
    ----------
    x : ndarray
        First variable.
    y : ndarray
        Second variable.
    n_bins : int
        Number of bins for discretization.

    Returns
    -------
    su : float
        Symmetrical uncertainty (between 0 and 1).
    """
    # Discretize if continuous
    if np.issubdtype(x.dtype, np.floating):
        x_discrete = _discretize_continuous(x, n_bins)
    else:
        x_discrete = x.astype(int)

    if np.issubdtype(y.dtype, np.floating):
        y_discrete = _discretize_continuous(y, n_bins)
    else:
        y_discrete = y.astype(int)

    h_x = _entropy(x_discrete)
    h_y = _entropy(y_discrete)

    if h_x + h_y == 0:
        return 0.0

    # I(X; Y) = H(Y) - H(Y|X)
    h_y_given_x = _conditional_entropy(y_discrete, x_discrete)
    mutual_info = h_y - h_y_given_x

    return 2 * mutual_info / (h_x + h_y)

@feature_selector(tags=["subset", "filter", "correlation"], version="1.0.0")
class CFSSelector(FeatureSelector, SelectorMixin):
    """Correlation-based Feature Selection (CFS).

    CFS evaluates subsets of features by considering the individual predictive ability 
    of each feature along with the degree of redundancy between them.

    Overview
    --------
    The merit of a feature subset :math:`S` is computed as:

    .. math::
        \\text{merit}(S) = \\frac{k \\bar{r}_{cf}}{\\sqrt{k + k(k-1)\\bar{r}_{ff}}}

    where:
    - :math:`k` is the number of features in subset :math:`S`.
    - :math:`\\bar{r}_{cf}` is the average feature-class correlation.
    - :math:`\\bar{r}_{ff}` is the average feature-feature inter-correlation.

    CFS prefers subsets that are highly correlated with the class while having 
    low inter-correlation.

    Parameters
    ----------
    n_bins : int, default=10
        Number of bins for discretizing continuous features before computing 
        symmetrical uncertainty.

    search_method : {"best_first", "greedy_forward"}, default="best_first"
        Search method for finding the subset with maximum merit:
        - ``"best_first"``: Best-first search with backtracking.
        - ``"greedy_forward"``: Simple forward greedy search.

    search_termination : int, default=5
        For ``"best_first"``: number of non-improving nodes before terminating.

    locally_predictive : bool, default=True
        If True, include locally predictive attributes (features that correlate 
        more with the class than with any already selected feature).

    Attributes
    ----------
    selected_features_ : np.ndarray
        Indices of the selected features.

    merit_ : float
        The CFS merit score of the final selected subset.

    Notes
    -----
    **When to use:**
    - When you want a fast, filter-based subset selection that handles redundancy.
    - When you have many features and want to reduce them without training a model.

    **Limitations:**
    - Only handles linear/monotone relationships via symmetrical uncertainty.
    - May struggle with complex non-linear feature interactions that a wrapper 
      method could find.

    References
    ----------
    .. [Hall1998] Hall, M. A. (1998). **Correlation-based Feature Subset Selection 
           for Machine Learning.** *PhD Thesis*, University of Waikato.

    See Also
    --------
    :class:`~tuiml.features.selection.GenericUnivariateSelector` : Univariate filter.
    :class:`~tuiml.features.selection.WrapperSelector` : Model-based subset selection.

    Examples
    --------
    Basic usage with search method configuration:

    >>> from tuiml.features.selection import CFSSelector
    >>> import numpy as np
    >>> X, y = np.random.randn(50, 10), np.random.randint(0, 2, 50)
    >>> selector = CFSSelector(search_method="best_first", search_termination=3)
    >>> X_new = selector.fit_transform(X, y)
    >>> print(f"Selected: {selector.selected_features_}")
    >>> print(f"Merit: {selector.merit_:.4f}")
    """

    def __init__(
        self,
        n_bins: int = 10,
        search_method: str = "best_first",
        search_termination: int = 5,
        locally_predictive: bool = True
    ):
        super().__init__()
        self.n_bins = n_bins
        self.search_method = search_method
        self.search_termination = search_termination
        self.locally_predictive = locally_predictive

        self.selected_features_: Optional[np.ndarray] = None
        self.merit_: Optional[float] = None
        self._correlation_matrix: Optional[np.ndarray] = None
        self._class_correlations: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "CFSSelector":
        """
        Fit the CFS selector.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : CFSSelector
            The fitted selector.
        """
        if y is None:
            raise ValueError("CFSSelector requires target values (y)")

        X = _ensure_numpy(X)
        y = _ensure_numpy(y)

        n_features = X.shape[1]
        self._n_features_in = n_features

        # Precompute correlation matrix and class correlations
        self._precompute_correlations(X, y)

        # Search for best subset
        if self.search_method == "best_first":
            best_subset = self._best_first_search(n_features)
        else:  # greedy_forward
            best_subset = self._greedy_forward_search(n_features)

        # Add locally predictive attributes if requested
        if self.locally_predictive:
            best_subset = self._add_locally_predictive(best_subset, n_features)

        # Store results
        self._selected_indices = np.array(sorted(best_subset), dtype=int)
        self.selected_features_ = self._selected_indices
        self.merit_ = self._evaluate_subset(best_subset)
        self._feature_scores = self._class_correlations.copy()
        self._is_fitted = True

        return self

    def _precompute_correlations(self, X: np.ndarray, y: np.ndarray) -> None:
        """Precompute feature-class and feature-feature correlation matrices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,)
            Target values.
        """
        n_features = X.shape[1]

        # Feature-class correlations using symmetrical uncertainty
        self._class_correlations = np.zeros(n_features)
        for i in range(n_features):
            self._class_correlations[i] = _symmetrical_uncertainty(
                X[:, i], y, self.n_bins
            )

        # Feature-feature correlations (lazy computation)
        # Use -1 as sentinel for uncomputed
        self._correlation_matrix = np.full((n_features, n_features), -1.0)
        self._X = X  # Store for lazy computation

    def _get_feature_correlation(self, i: int, j: int) -> float:
        """Get or lazily compute symmetrical uncertainty between features i and j.

        Parameters
        ----------
        i : int
            Index of the first feature.
        j : int
            Index of the second feature.

        Returns
        -------
        corr : float
            Symmetrical uncertainty correlation value.
        """
        if i == j:
            return 1.0

        # Ensure i < j for symmetric access
        if i > j:
            i, j = j, i

        if self._correlation_matrix[i, j] == -1:
            corr = _symmetrical_uncertainty(
                self._X[:, i], self._X[:, j], self.n_bins
            )
            self._correlation_matrix[i, j] = corr
            self._correlation_matrix[j, i] = corr

        return self._correlation_matrix[i, j]

    def _evaluate_subset(self, subset: Set[int]) -> float:
        """
        Evaluate the merit of a feature subset.

        merit(S) = k * mean(r_cf) / sqrt(k + k*(k-1) * mean(r_ff))
        """
        if len(subset) == 0:
            return 0.0

        k = len(subset)
        subset_list = list(subset)

        # Sum of feature-class correlations
        sum_cf = sum(self._class_correlations[f] for f in subset_list)

        # Sum of feature-feature correlations
        sum_ff = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                sum_ff += self._get_feature_correlation(subset_list[i], subset_list[j])

        # Mean correlations
        mean_cf = sum_cf / k
        mean_ff = sum_ff / (k * (k - 1) / 2) if k > 1 else 0

        # CFS merit
        denominator = np.sqrt(k + k * (k - 1) * mean_ff)
        if denominator == 0:
            return 0.0

        return k * mean_cf / denominator

    def _greedy_forward_search(self, n_features: int) -> Set[int]:
        """Select features by greedily adding the best-improving feature.

        Parameters
        ----------
        n_features : int
            Total number of available features.

        Returns
        -------
        selected : set of int
            Indices of the selected feature subset.
        """
        selected = set()
        remaining = set(range(n_features))
        best_merit = 0.0

        while remaining:
            best_feature = None
            best_new_merit = best_merit

            for feature in remaining:
                test_subset = selected | {feature}
                merit = self._evaluate_subset(test_subset)

                if merit > best_new_merit:
                    best_new_merit = merit
                    best_feature = feature

            if best_feature is not None and best_new_merit > best_merit:
                selected.add(best_feature)
                remaining.remove(best_feature)
                best_merit = best_new_merit
            else:
                break

        return selected

    def _best_first_search(self, n_features: int) -> Set[int]:
        """Search for the best feature subset using best-first search with backtracking.

        Parameters
        ----------
        n_features : int
            Total number of available features.

        Returns
        -------
        best_subset : set of int
            Indices of the best feature subset found.
        """
        # Priority queue: list of (merit, subset) tuples
        open_set = []
        closed_set = set()

        # Start with empty set
        initial = frozenset()
        open_set.append((0.0, initial))

        best_subset = initial
        best_merit = 0.0
        stale_count = 0

        while open_set and stale_count < self.search_termination:
            # Get best node
            open_set.sort(key=lambda x: x[0], reverse=True)
            current_merit, current_set = open_set.pop(0)

            # Convert to hashable form for closed set check
            if current_set in closed_set:
                continue

            closed_set.add(current_set)

            # Update best if improved
            if current_merit > best_merit:
                best_merit = current_merit
                best_subset = current_set
                stale_count = 0
            else:
                stale_count += 1

            # Generate successors (add one feature)
            for feature in range(n_features):
                if feature not in current_set:
                    new_set = current_set | {feature}
                    if new_set not in closed_set:
                        merit = self._evaluate_subset(set(new_set))
                        open_set.append((merit, new_set))

        return set(best_subset)

    def _add_locally_predictive(self, subset: Set[int], n_features: int) -> Set[int]:
        """
        Add locally predictive attributes.

        A feature is locally predictive if it has higher correlation
        with the class than with any feature already in the subset.
        """
        if len(subset) == 0:
            return subset

        result = set(subset)

        # Sort remaining features by class correlation
        remaining = set(range(n_features)) - result
        sorted_remaining = sorted(
            remaining,
            key=lambda f: self._class_correlations[f],
            reverse=True
        )

        for feature in sorted_remaining:
            class_corr = self._class_correlations[feature]

            # Check if feature has lower correlation with any selected feature
            is_locally_predictive = True
            for selected in result:
                feature_corr = self._get_feature_correlation(feature, selected)
                if feature_corr >= class_corr:
                    is_locally_predictive = False
                    break

            if is_locally_predictive:
                result.add(feature)

        return result

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return feature-class correlation scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (unused, scores are precomputed).
        y : ndarray of shape (n_samples,)
            Target values (unused, scores are precomputed).

        Returns
        -------
        scores : ndarray of shape (n_features,)
            Symmetrical uncertainty between each feature and the class.
        """
        return self._class_correlations

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X by selecting the CFS-chosen feature subset.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_selected_features)
            Data with only the selected features.
        """
        self._check_is_fitted()
        X = _ensure_numpy(X)
        if len(self._selected_indices) == 0:
            return X[:, :0]
        return X[:, self._selected_indices]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_bins": {
                "type": "integer",
                "default": 10,
                "minimum": 2,
                "description": "Number of bins for discretizing continuous features"
            },
            "search_method": {
                "type": "string",
                "enum": ["best_first", "greedy_forward"],
                "default": "best_first",
                "description": "Search method for finding optimal subset"
            },
            "search_termination": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "description": "Non-improving nodes before termination (best_first only)"
            },
            "locally_predictive": {
                "type": "boolean",
                "default": True,
                "description": "Include locally predictive attributes"
            }
        }

@feature_selector(tags=["subset", "wrapper", "cross-validation"], version="1.0.0")
class WrapperSelector(FeatureSelector, SelectorMixin):
    """Wrapper-based Feature Selection using cross-validation.

    Evaluates feature subsets by training a learning algorithm and using its 
    performance (e.g., accuracy) as the merit of the subset.

    Overview
    --------
    Wrapper selection is a comprehensive approach that considers feature interactions 
    by treating the learning algorithm as a "black box" to score subsets. It uses 
    cross-validation to provide a robust performance estimate.

    Search Strategies
    -----------------
    - ``"greedy_forward"``: Start with no features and add one at a time.
    - ``"greedy_backward"``: Start with all features and remove one at a time.
    - ``"best_first"``: Search through subsets using priority queue with backtracking.

    Parameters
    ----------
    estimator : object
        A classifier or regressor with ``fit`` and ``predict`` methods. 
        Must follow the scikit-learn estimator interface.

    cv : int, default=5
        Number of cross-validation folds for evaluating each subset.

    scoring : {"accuracy", "f1", "precision", "recall"}, default="accuracy"
        The performance metric used to rank subsets.

    search_method : {"greedy_forward", "greedy_backward", "best_first"}, default="greedy_forward"
        The algorithm used to explore the subset space.

    search_termination : int, default=5
        For ``"best_first"``: how many non-improving expansions to allow.

    random_state : int, optional
        Random seed for cross-validation shuffling.

    Attributes
    ----------
    selected_features_ : np.ndarray
        Indices of the features in the best subset found.

    cv_score_ : float
        The cross-validation score achieved by the final selected subset.

    Notes
    -----
    **When to use:**
    - When you want to find the absolute best subset for a specific model.
    - When feature interactions are crucial (e.g., XOR-like problems).

    **Limitations:**
    - Computationally very expensive, especially with many features or slow models.
    - High risk of overfitting to the validation set if the dataset is small.

    References
    ----------
    .. [Kohavi1997] Kohavi, R. and John, G. (1997). **Wrappers for feature subset 
           selection.** *Artificial Intelligence*, 97(1-2), 273-324.

    See Also
    --------
    :class:`~tuiml.features.selection.CFSSelector` : Fast correlation-based filter.
    :class:`~tuiml.features.selection.SequentialFeatureSelector` : Pure greedy wrapper.

    Examples
    --------
    Using a Decision Tree with forward selection:

    >>> from tuiml.features.selection import WrapperSelector
    >>> from tuiml.algorithms.trees import C45TreeClassifier
    >>> import numpy as np
    >>> X, y = np.random.randn(20, 6), np.random.randint(0, 2, 20)
    >>> selector = WrapperSelector(
    ...     estimator=C45TreeClassifier(),
    ...     cv=3,
    ...     search_method="greedy_forward"
    ... )
    >>> X_new = selector.fit_transform(X, y)
    >>> print(f"Selected: {selector.selected_features_}")
    >>> print(f"Best CV Score: {selector.cv_score_:.4f}")
    """

    def __init__(
        self,
        estimator: Any,
        cv: int = 5,
        scoring: str = "accuracy",
        search_method: str = "greedy_forward",
        search_termination: int = 5,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.search_method = search_method
        self.search_termination = search_termination
        self.random_state = random_state

        self.selected_features_: Optional[np.ndarray] = None
        self.cv_score_: Optional[float] = None
        self._feature_scores: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "WrapperSelector":
        """
        Fit the wrapper selector.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : WrapperSelector
            The fitted selector.
        """
        if y is None:
            raise ValueError("WrapperSelector requires target values (y)")

        X = _ensure_numpy(X)
        y = _ensure_numpy(y)

        n_samples, n_features = X.shape
        self._n_features_in = n_features

        # Store data for evaluation
        self._X = X
        self._y = y

        # Search for best subset
        if self.search_method == "greedy_forward":
            best_subset = self._greedy_forward_search(n_features)
        elif self.search_method == "greedy_backward":
            best_subset = self._greedy_backward_search(n_features)
        else:  # best_first
            best_subset = self._best_first_search(n_features)

        # Store results
        self._selected_indices = np.array(sorted(best_subset), dtype=int)
        self.selected_features_ = self._selected_indices
        self.cv_score_ = self._evaluate_subset(best_subset) if best_subset else 0.0

        # Compute individual feature scores for compatibility
        self._feature_scores = self._compute_individual_scores(X, y)

        self._is_fitted = True
        return self

    def _evaluate_subset(self, subset: Set[int]) -> float:
        """
        Evaluate a feature subset using cross-validation.

        Parameters
        ----------
        subset : set of int
            Feature indices to evaluate.

        Returns
        -------
        score : float
            Cross-validation score.
        """
        if len(subset) == 0:
            return 0.0

        subset_list = list(subset)
        X_subset = self._X[:, subset_list]

        # Stratified k-fold cross-validation
        n_samples = len(self._y)
        scores = []

        # Create fold indices
        rng = np.random.RandomState(self.random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        fold_size = n_samples // self.cv

        for fold in range(self.cv):
            start = fold * fold_size
            end = start + fold_size if fold < self.cv - 1 else n_samples

            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])

            X_train, X_test = X_subset[train_idx], X_subset[test_idx]
            y_train, y_test = self._y[train_idx], self._y[test_idx]

            # Clone estimator and fit
            try:
                from copy import deepcopy
                clf = deepcopy(self.estimator)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # Compute score
                score = self._compute_score(y_test, y_pred)
                scores.append(score)
            except Exception:
                scores.append(0.0)

        return np.mean(scores)

    def _compute_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the selected scoring metric from true and predicted labels.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True class labels.
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.

        Returns
        -------
        score : float
            Score value for the configured scoring metric.
        """
        if self.scoring == "accuracy":
            return np.mean(y_true == y_pred)
        elif self.scoring == "f1":
            # Simplified F1 for binary/multiclass
            return self._f1_score(y_true, y_pred)
        elif self.scoring == "precision":
            return self._precision_score(y_true, y_pred)
        elif self.scoring == "recall":
            return self._recall_score(y_true, y_pred)
        else:
            return np.mean(y_true == y_pred)

    def _f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute macro-averaged F1 score across all classes.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True class labels.
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.

        Returns
        -------
        f1 : float
            Macro-averaged F1 score.
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        f1_scores = []

        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)

        return np.mean(f1_scores)

    def _precision_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute macro-averaged precision across all classes.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True class labels.
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.

        Returns
        -------
        precision : float
            Macro-averaged precision.
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        precisions = []

        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precisions.append(precision)

        return np.mean(precisions)

    def _recall_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute macro-averaged recall across all classes.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            True class labels.
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.

        Returns
        -------
        recall : float
            Macro-averaged recall.
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        recalls = []

        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(recall)

        return np.mean(recalls)

    def _greedy_forward_search(self, n_features: int) -> Set[int]:
        """Select features by greedily adding the best-improving feature.

        Parameters
        ----------
        n_features : int
            Total number of available features.

        Returns
        -------
        selected : set of int
            Indices of the selected feature subset.
        """
        selected = set()
        remaining = set(range(n_features))
        best_score = 0.0

        while remaining:
            best_feature = None
            best_new_score = best_score

            for feature in remaining:
                test_subset = selected | {feature}
                score = self._evaluate_subset(test_subset)

                if score > best_new_score:
                    best_new_score = score
                    best_feature = feature

            if best_feature is not None and best_new_score > best_score:
                selected.add(best_feature)
                remaining.remove(best_feature)
                best_score = best_new_score
            else:
                break

        return selected

    def _greedy_backward_search(self, n_features: int) -> Set[int]:
        """Eliminate features by greedily removing the least-impactful feature.

        Parameters
        ----------
        n_features : int
            Total number of available features.

        Returns
        -------
        selected : set of int
            Indices of the remaining feature subset.
        """
        selected = set(range(n_features))
        best_score = self._evaluate_subset(selected)

        while len(selected) > 1:
            worst_feature = None
            best_new_score = 0.0

            for feature in selected:
                test_subset = selected - {feature}
                score = self._evaluate_subset(test_subset)

                if score >= best_new_score:
                    best_new_score = score
                    worst_feature = feature

            if worst_feature is not None and best_new_score >= best_score:
                selected.remove(worst_feature)
                best_score = best_new_score
            else:
                break

        return selected

    def _best_first_search(self, n_features: int) -> Set[int]:
        """Search for the best feature subset using best-first search with backtracking.

        Parameters
        ----------
        n_features : int
            Total number of available features.

        Returns
        -------
        best_subset : set of int
            Indices of the best feature subset found.
        """
        open_set = []
        closed_set = set()

        initial = frozenset()
        open_set.append((0.0, initial))

        best_subset = initial
        best_score = 0.0
        stale_count = 0

        while open_set and stale_count < self.search_termination:
            open_set.sort(key=lambda x: x[0], reverse=True)
            current_score, current_set = open_set.pop(0)

            if current_set in closed_set:
                continue

            closed_set.add(current_set)

            if current_score > best_score:
                best_score = current_score
                best_subset = current_set
                stale_count = 0
            else:
                stale_count += 1

            # Generate successors (add one feature)
            for feature in range(n_features):
                if feature not in current_set:
                    new_set = current_set | {feature}
                    if new_set not in closed_set:
                        score = self._evaluate_subset(set(new_set))
                        open_set.append((score, new_set))

        return set(best_subset)

    def _compute_individual_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute per-feature CV scores by evaluating each feature individually.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        scores : ndarray of shape (n_features,)
            Cross-validation score for each feature evaluated alone.
        """
        n_features = X.shape[1]
        scores = np.zeros(n_features)

        for i in range(n_features):
            scores[i] = self._evaluate_subset({i})

        return scores

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return precomputed individual feature CV scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (unused, scores are precomputed).
        y : ndarray of shape (n_samples,)
            Target values (unused, scores are precomputed).

        Returns
        -------
        scores : ndarray of shape (n_features,)
            Individual feature cross-validation scores.
        """
        return self._feature_scores

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X by selecting the wrapper-chosen feature subset.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_selected_features)
            Data with only the selected features.
        """
        self._check_is_fitted()
        X = _ensure_numpy(X)
        if len(self._selected_indices) == 0:
            return X[:, :0]
        return X[:, self._selected_indices]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "estimator": {
                "type": "object",
                "description": "Classifier with fit/predict methods"
            },
            "cv": {
                "type": "integer",
                "default": 5,
                "minimum": 2,
                "description": "Number of cross-validation folds"
            },
            "scoring": {
                "type": "string",
                "enum": ["accuracy", "f1", "precision", "recall"],
                "default": "accuracy",
                "description": "Scoring metric"
            },
            "search_method": {
                "type": "string",
                "enum": ["greedy_forward", "greedy_backward", "best_first"],
                "default": "greedy_forward",
                "description": "Search method for finding optimal subset"
            },
            "search_termination": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "description": "Non-improving nodes before termination (best_first only)"
            },
            "random_state": {
                "type": "integer",
                "description": "Random state for reproducibility"
            }
        }
