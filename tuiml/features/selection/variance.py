"""
Variance-based feature selection.

This module provides feature selectors that remove low-variance features.
"""

import numpy as np
from typing import Dict, Optional, Any

from tuiml.base.features import FeatureSelector, feature_selector
from ._base import SelectorMixin, _ensure_numpy

@feature_selector(tags=["variance", "filter", "unsupervised"], version="1.0.0")
class VarianceThresholdSelector(FeatureSelector, SelectorMixin):
    """Feature selector that removes all low-variance features.

    This is a simple baseline approach for feature selection. It removes all features 
    whose variance doesn't meet a threshold. By default, it removes all zero-variance 
    features, i.e., features that have the same value in all samples.

    Overview
    --------
    The selector computes the variance of each feature across all samples. If the 
    variance of a feature is less than or equal to the specified threshold, the 
    feature is removed. This method is unsupervised as it does not look at the 
    target values.

    Theory
    ------
    For each feature :math:`X_j`, the variance is computed as:

    .. math::
        \\sigma^2(X_j) = \\frac{1}{n} \\sum_{i=1}^n (x_{ij} - \\bar{x}_j)^2

    If :math:`\\sigma^2(X_j) \\le \\tau`, where :math:`\\tau` is the threshold, 
    feature :math:`j` is discarded.

    Parameters
    ----------
    threshold : float, default=0.0
        Features with a variance lower than or equal to this threshold will be 
        removed. The default is to keep all features with non-zero variance.

    Attributes
    ----------
    variances_ : np.ndarray of shape (n_features,)
        Variance of each feature computed from the training data.

    Notes
    -----
    **When to use:**
    - As a first step in feature selection to remove constant or near-constant features.
    - When you have a very large number of features and want a computationally 
      inexpensive way to reduce dimensionality.
    - In unsupervised scenarios where target labels are unavailable.

    **Limitations:**
    - Does not take into account the relationship between features and the target.
    - Does not handle feature redundancy (redundant features with high variance 
      will both be kept).

    See Also
    --------
    :class:`~tuiml.features.selection.GenericUnivariateSelector` : Configurable univariate selector.
    :class:`~tuiml.features.selection.CFSSelector` : Correlation-based feature selection.

    Examples
    --------
    Basic usage for removing zero-variance features:

    >>> from tuiml.features.selection import VarianceThresholdSelector
    >>> import numpy as np
    >>> X = np.array([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    >>> selector = VarianceThresholdSelector()
    >>> X_new = selector.fit_transform(X)
    >>> print(X_new)
    [[2 0]
     [1 4]
     [1 1]]

    Remove features with variance below 0.1:

    >>> selector = VarianceThresholdSelector(threshold=0.1)
    >>> X_new = selector.fit_transform(X)
    >>> print(X_new)
    [[2 0]
     [1 4]
     [1 1]]
    """

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold
        self.variances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "VarianceThresholdSelector":
        """
        Learn variances from X.

        Note: y is ignored (unsupervised feature selection).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : VarianceThresholdSelector
            The fitted selector.
        """
        X = _ensure_numpy(X)

        if self.threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {self.threshold}")

        self._n_features_in = X.shape[1]

        # Compute variance of each feature
        # Use nanvar to handle missing values
        self.variances_ = np.nanvar(X, axis=0)
        self._feature_scores = self.variances_

        # Select features with variance above threshold
        self._selected_indices = np.where(self.variances_ > self.threshold)[0]
        self._selected_indices = np.sort(self._selected_indices)
        self._is_fitted = True

        return self

    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return precomputed feature variance scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (unused, variances are precomputed).
        y : ndarray of shape (n_samples,)
            Target values (unused, unsupervised selector).

        Returns
        -------
        variances : ndarray of shape (n_features,)
            Variance of each feature.
        """
        return self.variances_

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
            Data with low-variance features removed.
        """
        self._check_is_fitted()
        X = _ensure_numpy(X)
        if len(self._selected_indices) == 0:
            return X[:, :0]
        return X[:, self._selected_indices]

    def get_support(self, indices: bool = False) -> np.ndarray:
        """
        Get a mask or indices of selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, return indices; if False, return boolean mask.

        Returns
        -------
        support : ndarray
            Boolean mask or indices of selected features.
        """
        self._check_is_fitted()
        if indices:
            return self._selected_indices.copy()
        else:
            mask = np.zeros(self._n_features_in, dtype=bool)
            mask[self._selected_indices] = True
            return mask

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "threshold": {
                "type": "number",
                "default": 0.0,
                "minimum": 0,
                "description": "Variance threshold. Features with variance <= threshold are removed."
            }
        }
