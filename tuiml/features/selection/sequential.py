"""
Sequential feature selection methods.

This module provides sequential/greedy feature selection methods that
iteratively add or remove features based on model performance.
- SequentialFeatureSelector: GreedyStepwise.java
- BestFirstSelector: BestFirst.java
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Literal
from collections import deque
import warnings

from tuiml.base.features import FeatureSelector, feature_selector
from ._base import SelectorMixin, _ensure_numpy

def _cross_val_score(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    random_state: Optional[int] = None
) -> float:
    """
    Simple cross-validation scoring function.

    Parameters
    ----------
    estimator : estimator object
        Estimator with fit/predict methods.
    X : ndarray
        Feature matrix.
    y : ndarray
        Target values.
    cv : int
        Number of folds.
    random_state : int, optional
        Random seed.

    Returns
    -------
    score : float
        Mean cross-validated accuracy.
    """
    n_samples = X.shape[0]
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_size = n_samples // cv
    scores = []

    for fold in range(cv):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < cv - 1 else n_samples

        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone the estimator (simple approach)
        try:
            params = {k: v for k, v in estimator.get_params().items()
                      if not k.endswith('_')}
            clf = estimator.__class__(**params)
        except (AttributeError, TypeError):
            clf = estimator

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(np.mean(y_pred == y_test))

    return np.mean(scores)

@feature_selector(tags=["wrapper", "sequential", "greedy"], version="1.0.0")
class SequentialFeatureSelector(FeatureSelector, SelectorMixin):
    """Sequential feature selector (forward or backward selection).

    Performs a greedy search through the space of feature subsets by iteratively 
    adding (forward) or removing (backward) features based on a model's performance.

    Overview
    --------
    Sequential feature selection is a wrapper method that evaluates feature subsets 
    using a learning algorithm. It starts with an empty set of features (forward) or 
    the full set (backward) and greedily modifies the subset to maximize a performance 
    metric.

    Process
    -------
    1. **Forward Selection**: Start with zero features. In each step, evaluate all 
       features not yet in the set. Add the one that improves the cross-validated 
       score the most.
    2. **Backward Selection**: Start with all features. In each step, try removing 
       each feature. Remove the one whose absence results in the highest score.

    The process continues until the desired number of features is reached or no 
    improvement above ``tol`` is found.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with ``fit`` and ``predict`` methods.

    n_features_to_select : int, float, or "auto", default="auto"
        Number of features to select:
        - ``int``: Select exactly this many features.
        - ``float``: Select this fraction of total features (0 < x < 1).
        - ``"auto"``: Stop when the score doesn't improve by at least ``tol``.

    direction : {"forward", "backward"}, default="forward"
        Search direction. Forward adds features, backward removes them.

    scoring : callable, optional
        Scoring function. If None, uses accuracy for classification.

    cv : int, default=5
        Number of cross-validation folds.

    tol : float, default=0.0
        Tolerance for improvement. Selection stops if the score doesn't improve 
        by at least ``tol``.

    random_state : int, optional
        Random seed for cross-validation splits.

    Attributes
    ----------
    n_features_to_select_ : int
        Actual number of features selected.

    support_ : np.ndarray of shape (n_features,)
        Boolean mask of selected features.

    Notes
    -----
    **Complexity:**
    - Roughly :math:`O(n_{features} \\cdot n_{select} \\cdot CV)` model fits.
    - More expensive than filter methods but captures feature interactions.

    **When to use:**
    - For small to medium datasets where feature interactions are important.
    - When you want to find a sparse set of highly predictive features.

    **Limitations:**
    - Computationally expensive for many features.
    - Greedy search may get stuck in local optima.

    References
    ----------
    .. [Ferri1994] Ferri, F. J., et al. (1994). **Comparative study of techniques for 
           large-scale feature selection.** *Pattern Recognition in Practice IV*, 
           pp. 403-413.

    See Also
    --------
    :class:`~tuiml.features.selection.BestFirstSelector` : Search with backtracking.
    :class:`~tuiml.features.selection.WrapperSelector` : Comprehensive wrapper selection.

    Examples
    --------
    Select 2 features forward using a simple estimator:

    >>> from tuiml.features.selection import SequentialFeatureSelector
    >>> from tuiml.algorithms.linear import LogisticRegression
    >>> import numpy as np
    >>> X, y = np.random.randn(20, 5), np.random.randint(0, 2, 20)
    >>> selector = SequentialFeatureSelector(
    ...     estimator=LogisticRegression(),
    ...     n_features_to_select=2,
    ...     direction='forward'
    ... )
    >>> X_new = selector.fit_transform(X, y)
    >>> print(selector.n_features_to_select_)
    2
    """

    def __init__(
        self,
        estimator: Any = None,
        n_features_to_select: Union[int, float, str] = "auto",
        direction: Literal["forward", "backward"] = "forward",
        scoring: Optional[Any] = None,
        cv: int = 5,
        tol: float = 0.0,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.tol = tol
        self.random_state = random_state

        self.n_features_to_select_: Optional[int] = None
        self.support_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SequentialFeatureSelector":
        """
        Fit the feature selector.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : SequentialFeatureSelector
            The fitted selector.
        """
        if y is None:
            raise ValueError("SequentialFeatureSelector requires target values (y)")

        if self.estimator is None:
            raise ValueError("estimator must be provided")

        X = _ensure_numpy(X)
        y = _ensure_numpy(y)

        n_features = X.shape[1]
        self._n_features_in = n_features

        # Determine target number of features
        if self.n_features_to_select == "auto":
            n_to_select = n_features  # Will be limited by convergence
        elif isinstance(self.n_features_to_select, float):
            n_to_select = max(1, int(n_features * self.n_features_to_select))
        else:
            n_to_select = min(self.n_features_to_select, n_features)

        # Initialize selected features
        if self.direction == "forward":
            selected = set()
            remaining = set(range(n_features))
            best_score = -np.inf
        else:  # backward
            selected = set(range(n_features))
            remaining = set()
            # Evaluate with all features
            best_score = self._evaluate_subset(X, y, list(selected))

        # Greedy search
        iteration = 0
        while True:
            iteration += 1

            if self.direction == "forward":
                # Try adding each remaining feature
                if len(remaining) == 0 or len(selected) >= n_to_select:
                    break

                best_feature = None
                best_new_score = best_score

                for feature in remaining:
                    test_features = list(selected | {feature})
                    score = self._evaluate_subset(X, y, test_features)

                    if score > best_new_score + self.tol:
                        best_new_score = score
                        best_feature = feature

                # Add best feature if improvement found
                if best_feature is not None:
                    selected.add(best_feature)
                    remaining.remove(best_feature)
                    best_score = best_new_score
                else:
                    # No improvement, stop
                    if self.n_features_to_select == "auto":
                        break
                    # Still need more features but no improvement
                    # Add the feature with highest score (may be worse)
                    best_feature = max(
                        remaining,
                        key=lambda f: self._evaluate_subset(X, y, list(selected | {f}))
                    )
                    selected.add(best_feature)
                    remaining.remove(best_feature)

            else:  # backward
                # Try removing each selected feature
                if len(selected) <= n_to_select:
                    if self.n_features_to_select != "auto":
                        break

                if len(selected) <= 1:
                    break

                best_feature = None
                best_new_score = best_score

                for feature in list(selected):
                    test_features = list(selected - {feature})
                    if len(test_features) == 0:
                        continue
                    score = self._evaluate_subset(X, y, test_features)

                    if score >= best_new_score - self.tol:
                        best_new_score = score
                        best_feature = feature

                # Remove best feature if no degradation
                if best_feature is not None:
                    selected.remove(best_feature)
                    best_score = best_new_score
                else:
                    # All removals cause degradation
                    if self.n_features_to_select == "auto":
                        break
                    # Still need to remove features
                    # Remove the feature with least degradation
                    best_feature = min(
                        selected,
                        key=lambda f: -self._evaluate_subset(X, y, list(selected - {f}))
                    )
                    selected.remove(best_feature)

        # Store results
        self._selected_indices = np.array(sorted(selected), dtype=int)
        self.n_features_to_select_ = len(self._selected_indices)
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[self._selected_indices] = True
        self._feature_scores = np.zeros(n_features)
        self._feature_scores[self._selected_indices] = 1.0
        self._is_fitted = True

        return self

    def _evaluate_subset(self, X: np.ndarray, y: np.ndarray, features: List[int]) -> float:
        """Evaluate a feature subset using cross-validation accuracy.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Full feature matrix.
        y : ndarray of shape (n_samples,)
            Target values.
        features : list of int
            Indices of features to include in the evaluation.

        Returns
        -------
        score : float
            Mean cross-validated accuracy for the given feature subset.
        """
        if len(features) == 0:
            return -np.inf

        X_subset = X[:, features]
        return _cross_val_score(
            self.estimator, X_subset, y,
            cv=self.cv, random_state=self.random_state
        )

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return binary feature scores indicating selected features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (unused, scores are precomputed).
        y : ndarray of shape (n_samples,)
            Target values (unused, scores are precomputed).

        Returns
        -------
        scores : ndarray of shape (n_features,)
            Binary scores (1.0 for selected, 0.0 for unselected).
        """
        return self._feature_scores

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X by selecting the sequentially chosen features.

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
        return X[:, self._selected_indices]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "estimator": {
                "type": "object",
                "description": "Estimator with fit/predict methods"
            },
            "n_features_to_select": {
                "type": ["integer", "number", "string"],
                "default": "auto",
                "description": "Number/fraction of features to select, or 'auto'"
            },
            "direction": {
                "type": "string",
                "enum": ["forward", "backward"],
                "default": "forward",
                "description": "Search direction"
            },
            "cv": {
                "type": "integer",
                "default": 5,
                "minimum": 2,
                "description": "Number of cross-validation folds"
            },
            "tol": {
                "type": "number",
                "default": 0.0,
                "minimum": 0,
                "description": "Tolerance for improvement"
            }
        }

@feature_selector(tags=["wrapper", "sequential", "best-first"], version="1.0.0")
class BestFirstSelector(FeatureSelector, SelectorMixin):
    """Best-first feature selector with backtracking.

    Performs a search through the space of feature subsets using a best-first 
    heuristic, which allows backtracking when a selected path leads to no improvement.

    Overview
    --------
    Best-first search uses a priority queue to explore the most promising feature 
    subsets first. Unlike greedy sequential selection, it can jump back to a 
    previously explored node if it looks more promising than the current expansion.

    Search Process
    --------------
    1. Maintain a list of "open" nodes (feature subsets) ranked by their CV score.
    2. Expand the best node by adding/removing one feature.
    3. If the best score hasn't improved for ``search_termination`` expansions, 
       stop and return the overall best subset.

    This strategy balances between greedy search and exhaustive exploration.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with ``fit`` and ``predict`` methods.

    direction : {"forward", "backward", "bidirectional"}, default="forward"
        Search direction:
        - ``"forward"``: Start with no features.
        - ``"backward"``: Start with all features.
        - ``"bidirectional"``: Can add or remove features at any step.

    search_termination : int, default=5
        Maximum number of consecutive non-improving expansions allowed before 
        the search terminates.

    cv : int, default=5
        Number of cross-validation folds for subset evaluation.

    random_state : int, optional
        Random seed for cross-validation splits.

    Attributes
    ----------
    n_features_selected_ : int
        Number of features in the best subset found.

    Notes
    -----
    **Complexity:**
    - Can be highly variable depending on ``search_termination``.
    - Generally more expensive than ``SequentialFeatureSelector`` but potentially 
      more effective at finding global optima.

    **When to use:**
    - When greedy selection fails to find a good subset.
    - When you have moderate number of features and computational budget.

    See Also
    --------
    :class:`~tuiml.features.selection.SequentialFeatureSelector` : Pure greedy selection.
    :class:`~tuiml.features.selection.CFSSelector` : Correlation-based filter.

    Examples
    --------
    Perform best-first search for 10 nodes:

    >>> from tuiml.features.selection import BestFirstSelector
    >>> from tuiml.algorithms.trees import C45TreeClassifier
    >>> import numpy as np
    >>> X, y = np.random.randn(20, 8), np.random.randint(0, 2, 20)
    >>> selector = BestFirstSelector(
    ...     estimator=C45TreeClassifier(),
    ...     direction='forward',
    ...     search_termination=3
    ... )
    >>> X_new = selector.fit_transform(X, y)
    >>> print(f"Selected {selector.n_features_selected_} features")
    """

    def __init__(
        self,
        estimator: Any = None,
        direction: Literal["forward", "backward", "bidirectional"] = "forward",
        search_termination: int = 5,
        cv: int = 5,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.estimator = estimator
        self.direction = direction
        self.search_termination = search_termination
        self.cv = cv
        self.random_state = random_state

        self.n_features_selected_: Optional[int] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BestFirstSelector":
        """
        Fit the feature selector using best-first search.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : BestFirstSelector
            The fitted selector.
        """
        if y is None:
            raise ValueError("BestFirstSelector requires target values (y)")

        if self.estimator is None:
            raise ValueError("estimator must be provided")

        X = _ensure_numpy(X)
        y = _ensure_numpy(y)

        n_features = X.shape[1]
        self._n_features_in = n_features

        # Initialize based on direction
        if self.direction == "forward":
            initial_set = frozenset()
        else:  # backward or bidirectional
            initial_set = frozenset(range(n_features))

        # Priority queue: (negative_score, feature_set)
        # Use negative score because we want max score
        open_set = []  # List of (score, feature_set) tuples
        closed_set = set()

        # Evaluate initial set
        initial_score = self._evaluate_subset(X, y, list(initial_set)) if initial_set else -np.inf
        open_set.append((initial_score, initial_set))

        best_set = initial_set
        best_score = initial_score
        stale_count = 0

        while open_set and stale_count < self.search_termination:
            # Get best node (highest score)
            open_set.sort(key=lambda x: x[0], reverse=True)
            current_score, current_set = open_set.pop(0)

            if current_set in closed_set:
                continue

            closed_set.add(current_set)

            # Update best if improved
            if current_score > best_score:
                best_score = current_score
                best_set = current_set
                stale_count = 0
            else:
                stale_count += 1

            # Generate neighbors
            neighbors = self._get_neighbors(current_set, n_features)

            for neighbor in neighbors:
                if neighbor not in closed_set:
                    score = self._evaluate_subset(X, y, list(neighbor))
                    open_set.append((score, neighbor))

        # Store results
        self._selected_indices = np.array(sorted(best_set), dtype=int)
        self.n_features_selected_ = len(self._selected_indices)
        self._feature_scores = np.zeros(n_features)
        self._feature_scores[self._selected_indices] = 1.0
        self._is_fitted = True

        return self

    def _get_neighbors(self, current_set: frozenset, n_features: int) -> List[frozenset]:
        """Generate neighboring feature sets by adding or removing one feature.

        Parameters
        ----------
        current_set : frozenset of int
            Current feature subset.
        n_features : int
            Total number of available features.

        Returns
        -------
        neighbors : list of frozenset
            Neighboring feature subsets based on the search direction.
        """
        neighbors = []

        if self.direction in ["forward", "bidirectional"]:
            # Add features not in set
            for i in range(n_features):
                if i not in current_set:
                    neighbors.append(current_set | {i})

        if self.direction in ["backward", "bidirectional"]:
            # Remove features from set
            for i in current_set:
                new_set = current_set - {i}
                if new_set:  # Don't allow empty set
                    neighbors.append(new_set)

        return neighbors

    def _evaluate_subset(self, X: np.ndarray, y: np.ndarray, features: List[int]) -> float:
        """Evaluate a feature subset using cross-validation accuracy.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Full feature matrix.
        y : ndarray of shape (n_samples,)
            Target values.
        features : list of int
            Indices of features to include in the evaluation.

        Returns
        -------
        score : float
            Mean cross-validated accuracy for the given feature subset.
        """
        if len(features) == 0:
            return -np.inf

        X_subset = X[:, features]
        return _cross_val_score(
            self.estimator, X_subset, y,
            cv=self.cv, random_state=self.random_state
        )

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return binary feature scores indicating selected features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (unused, scores are precomputed).
        y : ndarray of shape (n_samples,)
            Target values (unused, scores are precomputed).

        Returns
        -------
        scores : ndarray of shape (n_features,)
            Binary scores (1.0 for selected, 0.0 for unselected).
        """
        return self._feature_scores

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X by selecting the best-first-chosen features.

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
                "description": "Estimator with fit/predict methods"
            },
            "direction": {
                "type": "string",
                "enum": ["forward", "backward", "bidirectional"],
                "default": "forward",
                "description": "Search direction"
            },
            "search_termination": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "description": "Non-improving nodes before termination"
            },
            "cv": {
                "type": "integer",
                "default": 5,
                "minimum": 2,
                "description": "Number of cross-validation folds"
            }
        }
