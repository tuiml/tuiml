"""
Base classes and utilities for feature selection.

This module provides the foundation for all feature selection methods,
including mixins for common functionality and utility functions.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Any, Union, Tuple
import numpy as np

from tuiml.base.features import FeatureSelector, feature_selector

class SelectorMixin:
    """Mixin class providing common functionality for feature selectors.

    This mixin provides methods for getting the support mask or indices of selected 
    features, performing inverse transformations, and generating output feature names.
    It expects the following attributes to be set by the fitting method:
    - ``_selected_indices``: np.ndarray of selected feature indices.
    - ``_n_features_in``: int, number of input features.
    """

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get a mask or indices of the selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, returns an array of integer indices. 
            If False, returns a boolean mask.

        Returns
        -------
        support : np.ndarray
            An index or mask that selects the retained features.
        """
        self._check_is_fitted()
        if indices:
            return self._selected_indices.copy()
        else:
            mask = np.zeros(self._n_features_in, dtype=bool)
            mask[self._selected_indices] = True
            return mask

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the transformation operation.

        Reconstructs the original feature space by filling non-selected features 
        with zeros.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_selected_features)
            The selected feature matrix.

        Returns
        -------
        X_r : np.ndarray of shape (n_samples, n_input_features)
            Data projected back into the original feature space.
        """
        self._check_is_fitted()
        support = self.get_support()
        X_r = np.zeros((X.shape[0], support.shape[0]))
        X_r[:, support] = X
        return X_r

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : list of str, optional
            Input feature names. If None, uses default names.

        Returns
        -------
        feature_names_out : ndarray of str
            Names of the selected features.
        """
        self._check_is_fitted()
        if input_features is None:
            input_features = [f"x{i}" for i in range(self._n_features_in)]
        input_features = np.asarray(input_features)
        return input_features[self.get_support()]

def _check_feature_names(X, feature_names: Optional[List[str]] = None) -> Optional[List[str]]:
    """
    Check and return feature names from X or provided names.

    Parameters
    ----------
    X : array-like
        Input data.
    feature_names : list of str, optional
        Provided feature names.

    Returns
    -------
    feature_names : list of str or None
        Feature names if available.
    """
    if feature_names is not None:
        return list(feature_names)
    if hasattr(X, 'columns'):
        return list(X.columns)
    return None

def _ensure_numpy(X) -> np.ndarray:
    """
    Convert input to numpy array if needed.

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    X : ndarray
        Numpy array.
    """
    if hasattr(X, 'values'):
        return X.values
    return np.asarray(X)

def _validate_score_func(score_func: Callable) -> Callable:
    """
    Validate that score_func is callable.

    Parameters
    ----------
    score_func : callable
        Scoring function.

    Returns
    -------
    score_func : callable
        Validated scoring function.

    Raises
    ------
    TypeError
        If score_func is not callable.
    """
    if not callable(score_func):
        raise TypeError(
            f"score_func must be callable, got {type(score_func)}"
        )
    return score_func

def _get_scores_and_pvalues(score_func: Callable, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Call score function and extract scores and p-values.

    Parameters
    ----------
    score_func : callable
        Function taking (X, y) and returning scores or (scores, pvalues).
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target values.

    Returns
    -------
    scores : ndarray of shape (n_features,)
        Feature scores.
    pvalues : ndarray of shape (n_features,) or None
        P-values if returned by score_func, else None.
    """
    result = score_func(X, y)
    if isinstance(result, tuple) and len(result) == 2:
        scores, pvalues = result
        return np.asarray(scores), np.asarray(pvalues)
    else:
        return np.asarray(result), None

class GenericUnivariateSelector(FeatureSelector, SelectorMixin):
    """Univariate feature selector with configurable strategy.

    GenericUnivariateSelector allows to perform univariate feature selection with a 
    configurable strategy. It evaluates each feature individually and selects the 
    best ones based on the specified mode and parameter.

    Overview
    --------
    The selector works by:
    1. Computing a score for each feature using the provided ``score_func``.
    2. Optional: Computing p-values for each feature score.
    3. Selecting features based on the chosen ``mode`` and ``param``.

    Selection Modes
    ---------------
    - ``"k_best"``: Select the :math:`k` features with highest scores.
    - ``"percentile"``: Select top :math:`p` percentile of features.
    - ``"fpr"``: Select features with p-value below alpha (False Positive Rate).
    - ``"fdr"``: Select features based on Benjamini-Hochberg procedure (False Discovery Rate).
    - ``"fwe"``: Select features based on Bonferroni correction (Family-Wise Error rate).

    Parameters
    ----------
    score_func : callable, default=None
        Function taking (X, y) and returning (scores, pvalues) or scores. 
        Usually a statistical test like ``f_classif`` or ``mutual_info_classif``.

    mode : {"k_best", "percentile", "fpr", "fdr", "fwe"}, default="k_best"
        Strategy for Determining which features to keep.

    param : float or int, default=10
        Parameter for the selected mode:
        - For ``"k_best"``: number of features to select (int).
        - For ``"percentile"``: percentage of features to select (0-100 float).
        - For ``"fpr"``, ``"fdr"``, ``"fwe"``: alpha threshold (float).

    Attributes
    ----------
    scores_ : np.ndarray of shape (n_features,)
        Individual feature scores.

    pvalues_ : np.ndarray of shape (n_features,) or None
        P-values of feature scores, if returned by ``score_func``.

    Notes
    -----
    **When to use:**
    - When you want to perform statistical feature selection.
    - As a fast preprocessing step before more complex algorithms.
    - When you have a clear hypothesis about individual feature importance.

    **Limitations:**
    - Does not account for feature interactions (it's univariate).
    - Choice of ``score_func`` strongly impacts results.
    - Statistical tests may have assumptions (e.g., normality).

    See Also
    --------
    :class:`~tuiml.features.selection.VarianceThresholdSelector` : Remove low-variance features.
    :class:`~tuiml.features.selection.SequentialFeatureSelector` : Greedy feature selection.

    Examples
    --------
    Select top 2 features using a mock score function:

    >>> from tuiml.features.selection import GenericUnivariateSelector
    >>> import numpy as np
    >>> def mock_score(X, y): return np.array([0.1, 0.5, 0.2, 0.8])
    >>> X = np.random.randn(10, 4)
    >>> y = np.random.randint(0, 2, 10)
    >>> selector = GenericUnivariateSelector(score_func=mock_score, mode='k_best', param=2)
    >>> X_new = selector.fit_transform(X, y)
    >>> print(selector.get_support(indices=True))
    [1 3]
    """

    _selection_modes = {'k_best', 'percentile', 'fpr', 'fdr', 'fwe'}

    def __init__(
        self,
        score_func: Callable = None,
        mode: str = 'k_best',
        param: Union[int, float] = 10
    ):
        """Initialize GenericUnivariateSelector with scoring function and selection mode.

        Parameters
        ----------
        score_func : callable, default=None
            Function taking (X, y) and returning scores or (scores, pvalues).
        mode : str, default='k_best'
            Feature selection mode: 'k_best', 'percentile', 'fpr', 'fdr', 'fwe'.
        param : int or float, default=10
            Parameter for the selected mode.
        """
        super().__init__()
        self.score_func = score_func
        self.mode = mode
        self.param = param

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "GenericUnivariateSelector":
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
        self : GenericUnivariateSelector
            The fitted selector.
        """
        if y is None:
            raise ValueError("y is required for univariate feature selection")

        if self.mode not in self._selection_modes:
            raise ValueError(
                f"mode must be one of {self._selection_modes}, got {self.mode}"
            )

        X = _ensure_numpy(X)
        y = _ensure_numpy(y)

        self._n_features_in = X.shape[1]

        # Compute scores
        if self.score_func is None:
            raise ValueError("score_func must be provided")

        score_func = _validate_score_func(self.score_func)
        self.scores_, self.pvalues_ = _get_scores_and_pvalues(score_func, X, y)
        self._feature_scores = self.scores_

        # Select features based on mode
        self._selected_indices = self._get_selected_indices()
        self._selected_indices = np.sort(self._selected_indices)
        self._is_fitted = True

        return self

    def _get_selected_indices(self) -> np.ndarray:
        """Get indices of selected features based on mode and param.

        Returns
        -------
        indices : ndarray
            Indices of features selected by the current mode and param.
        """
        n_features = len(self.scores_)

        if self.mode == 'k_best':
            k = min(int(self.param), n_features)
            return np.argsort(self.scores_)[-k:]

        elif self.mode == 'percentile':
            percentile = max(0, min(100, self.param))
            k = max(1, int(n_features * percentile / 100))
            return np.argsort(self.scores_)[-k:]

        elif self.mode == 'fpr':
            if self.pvalues_ is None:
                raise ValueError("fpr mode requires p-values from score_func")
            return np.where(self.pvalues_ < self.param)[0]

        elif self.mode == 'fdr':
            if self.pvalues_ is None:
                raise ValueError("fdr mode requires p-values from score_func")
            # Benjamini-Hochberg procedure
            sorted_idx = np.argsort(self.pvalues_)
            sorted_pvalues = self.pvalues_[sorted_idx]
            n = len(sorted_pvalues)
            threshold = (np.arange(1, n + 1) / n) * self.param
            selected = sorted_pvalues <= threshold
            if np.any(selected):
                max_idx = np.max(np.where(selected)[0])
                return sorted_idx[:max_idx + 1]
            return np.array([], dtype=int)

        elif self.mode == 'fwe':
            if self.pvalues_ is None:
                raise ValueError("fwe mode requires p-values from score_func")
            # Bonferroni correction
            threshold = self.param / len(self.pvalues_)
            return np.where(self.pvalues_ < threshold)[0]

        return np.arange(n_features)

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
            Feature scores from the univariate scoring function.
        """
        return self.scores_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Select features from X.

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
            "mode": {
                "type": "string",
                "enum": ["k_best", "percentile", "fpr", "fdr", "fwe"],
                "default": "k_best",
                "description": "Feature selection mode"
            },
            "param": {
                "type": ["integer", "number"],
                "default": 10,
                "description": "Parameter for the selected mode"
            }
        }
