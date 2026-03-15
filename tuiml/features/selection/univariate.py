"""
Univariate feature selection methods.

This module provides univariate feature selectors that score each feature
independently and select the best ones based on the scores.
- SelectKBestSelector: Ranker.java with -N (numToSelect) option
- SelectPercentileSelector: Ranker.java with percentage-based selection
- SelectThresholdSelector: Ranker.java with -T (threshold) option
- SelectFprSelector: False positive rate threshold (p-value based)
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Any, Union

from tuiml.base.features import FeatureSelector, feature_selector
from ._base import SelectorMixin, _ensure_numpy, _validate_score_func, _get_scores_and_pvalues
from tuiml.evaluation.metrics import f_classif

@feature_selector(tags=["univariate", "filter", "ranking"], version="1.0.0")
class SelectKBestSelector(FeatureSelector, SelectorMixin):
    """Select features according to the k highest scores.

    Ranker-based feature selection that computes a univariate score for each feature 
    and selects the top :math:`k` performing features.

    Overview
    --------
    This selector evaluates each feature independently using a provided ``score_func``. 
    It is a fast filter method that provides a global ranking of feature importance 
    based on statistical tests.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking (X, y) and returning (scores, pvalues) or scores. 
        Usually a statistical test like ``f_classif``, ``chi2``, or ``mutual_info_classif``.

    k : int or "all", default=10
        Number of top features to select. If "all", no features are removed.

    Attributes
    ----------
    scores_ : np.ndarray of shape (n_features,)
        Individual feature scores.

    pvalues_ : np.ndarray of shape (n_features,) or None
        P-values of feature scores, if supported by ``score_func``.

    Notes
    -----
    **When to use:**
    - To identify the most statistically significant individual features.
    - As a baseline for more complex feature selection methods.
    - When you need a fast and interpretable way to reduce feature count.

    **Limitations:**
    - Only captures univariate importance; ignores feature interactions.
    - Does not handle redundant features (correlated features with high scores 
      will all be selected).

    See Also
    --------
    :class:`~tuiml.features.selection.SelectPercentileSelector` : Scale-based selection.
    :class:`~tuiml.features.selection.SelectThresholdSelector` : Score threshold selection.

    Examples
    --------
    Select top 5 features using information gain:

    >>> from tuiml.features.selection import SelectKBestSelector
    >>> from tuiml.evaluation.metrics import information_gain
    >>> import numpy as np
    >>> X, y = np.random.randn(20, 20), np.random.randint(0, 2, 20)
    >>> selector = SelectKBestSelector(score_func=information_gain, k=5)
    >>> X_new = selector.fit_transform(X, y)
    >>> print(selector.get_support(indices=True))
    """

    def __init__(
        self,
        score_func: Callable = None,
        k: Union[int, str] = 10
    ):
        # Note: Don't pass k to parent as it expects Optional[int]
        super().__init__(k=k if isinstance(k, int) else None)
        self.score_func = score_func if score_func is not None else f_classif
        self.k = k
        self.scores_: Optional[np.ndarray] = None
        self.pvalues_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SelectKBestSelector":
        """
        Fit the selector to the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : SelectKBestSelector
            The fitted selector.
        """
        if y is None:
            raise ValueError("SelectKBestSelector requires target values (y)")

        X = _ensure_numpy(X)
        y = _ensure_numpy(y)

        self._n_features_in = X.shape[1]

        # Validate score function
        score_func = _validate_score_func(self.score_func)

        # Compute scores
        self.scores_, self.pvalues_ = _get_scores_and_pvalues(score_func, X, y)
        self._feature_scores = self.scores_

        # Handle NaN scores
        self.scores_ = np.nan_to_num(self.scores_, nan=-np.inf)

        # Select top k features
        if self.k == "all":
            k = X.shape[1]
        else:
            k = min(self.k, X.shape[1])

        if k <= 0:
            self._selected_indices = np.array([], dtype=int)
        else:
            self._selected_indices = np.argsort(self.scores_)[-k:]

        self._selected_indices = np.sort(self._selected_indices)
        self._is_fitted = True

        return self

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return precomputed univariate feature scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (unused, scores are precomputed).
        y : ndarray of shape (n_samples,)
            Target values (unused, scores are precomputed).

        Returns
        -------
        scores : ndarray of shape (n_features,)
            Univariate feature scores.
        """
        return self.scores_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X by selecting the k highest-scoring features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_selected_features)
            Data with selected features.
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
            "score_func": {
                "type": "callable",
                "description": "Scoring function (X, y) -> scores or (scores, pvalues)"
            },
            "k": {
                "type": ["integer", "string"],
                "default": 10,
                "description": "Number of top features to select, or 'all'"
            }
        }

@feature_selector(tags=["univariate", "filter", "ranking"], version="1.0.0")
class SelectPercentileSelector(FeatureSelector, SelectorMixin):
    """Select features according to a percentile of the highest scores.

    A variant of :class:`SelectKBestSelector` that keeps a proportion of features 
    rather than a fixed number.

    Overview
    --------
    This selector is useful when you want to reduce dimensionality while keeping a 
    relative instead of absolute number of features across different datasets.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking (X, y) and returning (scores, pvalues) or scores.

    percentile : int, default=10
        Percent of features to keep (between 0 and 100).

    Attributes
    ----------
    scores_ : np.ndarray of shape (n_features,)
        Individual feature scores.

    pvalues_ : np.ndarray of shape (n_features,) or None
        P-values of feature scores.

    Examples
    --------
    Keep top 20% of features using Chi-Square test:

    >>> from tuiml.features.selection import SelectPercentileSelector
    >>> from tuiml.evaluation.metrics import chi2
    >>> import numpy as np
    >>> X, y = np.abs(np.random.randn(20, 50)), np.random.randint(0, 2, 20)
    >>> selector = SelectPercentileSelector(score_func=chi2, percentile=20)
    >>> X_new = selector.fit_transform(X, y)
    >>> print(X_new.shape[1])
    10
    """

    def __init__(
        self,
        score_func: Callable = None,
        percentile: int = 10
    ):
        super().__init__()
        self.score_func = score_func if score_func is not None else f_classif
        self.percentile = percentile
        self.scores_: Optional[np.ndarray] = None
        self.pvalues_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SelectPercentileSelector":
        """
        Fit the selector to the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : SelectPercentileSelector
            The fitted selector.
        """
        if y is None:
            raise ValueError("SelectPercentileSelector requires target values (y)")

        X = _ensure_numpy(X)
        y = _ensure_numpy(y)

        self._n_features_in = X.shape[1]

        # Validate percentile
        if not 0 <= self.percentile <= 100:
            raise ValueError(f"percentile must be between 0 and 100, got {self.percentile}")

        # Validate score function
        score_func = _validate_score_func(self.score_func)

        # Compute scores
        self.scores_, self.pvalues_ = _get_scores_and_pvalues(score_func, X, y)
        self._feature_scores = self.scores_

        # Handle NaN scores
        self.scores_ = np.nan_to_num(self.scores_, nan=-np.inf)

        # Select top percentile features
        n_features = X.shape[1]
        k = max(1, int(n_features * self.percentile / 100))

        if k <= 0 or self.percentile == 0:
            self._selected_indices = np.array([], dtype=int)
        else:
            self._selected_indices = np.argsort(self.scores_)[-k:]

        self._selected_indices = np.sort(self._selected_indices)
        self._is_fitted = True

        return self

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return precomputed univariate feature scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (unused, scores are precomputed).
        y : ndarray of shape (n_samples,)
            Target values (unused, scores are precomputed).

        Returns
        -------
        scores : ndarray of shape (n_features,)
            Univariate feature scores.
        """
        return self.scores_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X by selecting the top percentile of features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_selected_features)
            Data with selected features.
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
            "score_func": {
                "type": "callable",
                "description": "Scoring function (X, y) -> scores or (scores, pvalues)"
            },
            "percentile": {
                "type": "integer",
                "default": 10,
                "minimum": 0,
                "maximum": 100,
                "description": "Percent of features to keep"
            }
        }

@feature_selector(tags=["univariate", "filter", "ranking", "threshold"], version="1.0.0")
class SelectFprSelector(FeatureSelector, SelectorMixin):
    """Select features based on false positive rate threshold.

    Keeps features whose p-values are below a significance level ``alpha``, 
    thereby controlling the Probability of making a "False Positive" discovery.

    Overview
    --------
    This selector uses a statistical significance test (provided by ``score_func``) 
    to filter out features that are likely to be independent of the target variable.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking (X, y) and returning (scores, pvalues). 
        Must return p-values for thresholding.

    alpha : float, default=0.05
        Maximum p-value threshold for keeping a feature.

    Attributes
    ----------
    scores_ : np.ndarray of shape (n_features,)
        Individual feature scores.

    pvalues_ : np.ndarray of shape (n_features,)
        P-values associated with each feature.

    Notes
    -----
    **When to use:**
    - To select only features that have a statistically significant relationship 
      with the target.
    - When you want to control the False Positive Rate (FPR) of the selection process.

    See Also
    --------
    :class:`~tuiml.features.selection.GenericUnivariateSelector` : General modes.

    Examples
    --------
    Select features significant at 5% level:

    >>> from tuiml.features.selection import SelectFprSelector
    >>> from tuiml.evaluation.metrics import f_classif
    >>> import numpy as np
    >>> X, y = np.random.randn(50, 10), np.random.randint(0, 2, 50)
    >>> selector = SelectFprSelector(score_func=f_classif, alpha=0.05)
    >>> X_new = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        score_func: Callable = None,
        alpha: float = 0.05
    ):
        super().__init__()
        self.score_func = score_func if score_func is not None else f_classif
        self.alpha = alpha
        self.scores_: Optional[np.ndarray] = None
        self.pvalues_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SelectFprSelector":
        """
        Fit the selector to the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : SelectFprSelector
            The fitted selector.
        """
        if y is None:
            raise ValueError("SelectFprSelector requires target values (y)")

        X = _ensure_numpy(X)
        y = _ensure_numpy(y)

        self._n_features_in = X.shape[1]

        # Validate score function
        score_func = _validate_score_func(self.score_func)

        # Compute scores
        self.scores_, self.pvalues_ = _get_scores_and_pvalues(score_func, X, y)
        self._feature_scores = self.scores_

        if self.pvalues_ is None:
            raise ValueError("score_func must return p-values for SelectFprSelector")

        # Select features with p-value < alpha
        self._selected_indices = np.where(self.pvalues_ < self.alpha)[0]
        self._selected_indices = np.sort(self._selected_indices)
        self._is_fitted = True

        return self

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return precomputed univariate feature scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (unused, scores are precomputed).
        y : ndarray of shape (n_samples,)
            Target values (unused, scores are precomputed).

        Returns
        -------
        scores : ndarray of shape (n_features,)
            Univariate feature scores.
        """
        return self.scores_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X by selecting features with p-values below alpha.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_selected_features)
            Data with only statistically significant features.
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
            "score_func": {
                "type": "callable",
                "description": "Scoring function (X, y) -> (scores, pvalues)"
            },
            "alpha": {
                "type": "number",
                "default": 0.05,
                "minimum": 0,
                "maximum": 1,
                "description": "Maximum p-value threshold"
            }
        }

@feature_selector(tags=["univariate", "filter", "ranking", "threshold"], version="1.0.0")
class SelectThresholdSelector(FeatureSelector, SelectorMixin):
    """Select features based on score threshold.

    Keeps features with univariate scores above (or equal to) a specified threshold. 
    This corresponds to the Ranker search with the `-T` option in WEKA.

    Overview
    --------
    This selector filters features by their raw score value rather than rank or p-value. 
    It also allows ignoring specific feature indices during the ranking process.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking (X, y) and returning (scores, pvalues) or scores.

    threshold : float, default=0.0
        Minimum score threshold. Features with scores >= threshold are retained.

    ignore_features : list of int, optional
        Indices of features to ignore during ranking. These features will never 
        be selected, regardless of their score.

    Attributes
    ----------
    scores_ : np.ndarray of shape (n_features,)
        Individual feature scores.

    pvalues_ : np.ndarray of shape (n_features,) or None
        P-values of feature scores, if available.

    ranking_ : np.ndarray of shape (n_features,)
        Feature indices sorted by score in descending order.

    Notes
    -----
    **Comparing to FPR:**
    Unlike :class:`SelectFprSelector`, which thresholds p-values, this selector 
    thresholds the raw score values (e.g., Information Gain bits).

    Examples
    --------
    Select features with information gain >= 0.1:

    >>> from tuiml.features.selection import SelectThresholdSelector
    >>> from tuiml.evaluation.metrics import information_gain
    >>> import numpy as np
    >>> def mock_ig(X, y): return np.array([0.05, 0.15, 0.08, 0.12])
    >>> X = np.random.randn(10, 4)
    >>> y = np.random.randint(0, 2, 10)
    >>> selector = SelectThresholdSelector(score_func=mock_ig, threshold=0.1)
    >>> X_new = selector.fit_transform(X, y)
    >>> print(selector.get_support(indices=True))
    [1 3]
    """

    def __init__(
        self,
        score_func: Callable = None,
        threshold: float = 0.0,
        ignore_features: Optional[List[int]] = None
    ):
        super().__init__()
        self.score_func = score_func if score_func is not None else f_classif
        self.threshold = threshold
        self.ignore_features = ignore_features or []
        self.scores_: Optional[np.ndarray] = None
        self.pvalues_: Optional[np.ndarray] = None
        self.ranking_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SelectThresholdSelector":
        """
        Fit the selector to the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : SelectThresholdSelector
            The fitted selector.
        """
        if y is None:
            raise ValueError("SelectThresholdSelector requires target values (y)")

        X = _ensure_numpy(X)
        y = _ensure_numpy(y)

        n_features = X.shape[1]
        self._n_features_in = n_features

        # Validate score function
        score_func = _validate_score_func(self.score_func)

        # Compute scores
        self.scores_, self.pvalues_ = _get_scores_and_pvalues(score_func, X, y)
        self._feature_scores = self.scores_

        # Handle NaN scores
        scores_clean = np.nan_to_num(self.scores_, nan=-np.inf)

        # Create ranking (highest to lowest)
        self.ranking_ = np.argsort(scores_clean)[::-1]

        # Apply ignore_features mask
        ignore_set = set(self.ignore_features)
        valid_mask = np.array([i not in ignore_set for i in range(n_features)])

        # Select features with score >= threshold and not ignored
        threshold_mask = scores_clean >= self.threshold
        combined_mask = valid_mask & threshold_mask

        self._selected_indices = np.where(combined_mask)[0]
        self._selected_indices = np.sort(self._selected_indices)
        self._is_fitted = True

        return self

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return precomputed univariate feature scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (unused, scores are precomputed).
        y : ndarray of shape (n_samples,)
            Target values (unused, scores are precomputed).

        Returns
        -------
        scores : ndarray of shape (n_features,)
            Univariate feature scores.
        """
        return self.scores_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X by selecting features with scores above the threshold.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_selected_features)
            Data with only the features meeting the score threshold.
        """
        self._check_is_fitted()
        X = _ensure_numpy(X)
        if len(self._selected_indices) == 0:
            return X[:, :0]
        return X[:, self._selected_indices]

    def get_ranked_features(self) -> np.ndarray:
        """
        Get features sorted by score (highest to lowest).

        Returns
        -------
        ranking : ndarray of shape (n_features, 2)
            Array with columns [feature_index, score], sorted by score descending.
        """
        self._check_is_fitted()
        result = np.column_stack([
            self.ranking_,
            self.scores_[self.ranking_]
        ])
        return result

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "score_func": {
                "type": "callable",
                "description": "Scoring function (X, y) -> scores or (scores, pvalues)"
            },
            "threshold": {
                "type": "number",
                "default": 0.0,
                "description": "Minimum score threshold"
            },
            "ignore_features": {
                "type": "array",
                "items": {"type": "integer"},
                "default": [],
                "description": "Feature indices to ignore (WEKA -P option)"
            }
        }
