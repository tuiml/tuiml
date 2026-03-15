"""
Principal Component Analysis (PCAExtractor) for feature extraction.

This module provides PCAExtractor for dimensionality reduction and feature extraction.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union

from tuiml.base.features import FeatureExtractor, feature_extractor

@feature_extractor(tags=["extraction", "dimensionality_reduction", "pca"], version="1.0.0")
class PCAExtractor(FeatureExtractor):
    """Principal Component Analysis (PCAExtractor).

    Linear dimensionality reduction using Singular Value Decomposition (SVD) of the 
    data to project it to a lower dimensional space.

    Overview
    --------
    PCA identifies the axes (principal components) that maximize the variance in 
    the data. The first component accounts for the most variance, the second for 
    the next most, and so on.

    The algorithm works by:
    1. Centering the data (subtracting the mean).
    2. Computing the SVD: :math:`X = U \\Sigma V^T`.
    3. Selecting the first :math:`k` components from :math:`V^T`.
    4. Projecting the data onto these components.

    Parameters
    ----------
    n_components : int, float, or None, default=None
        Number of components to keep:
        - ``int``: Keep exactly this many components.
        - ``float`` (0 < x < 1): Select enough components to explain this proportion of variance.
        - ``None``: Keep all components (min(samples, features)).

    center : bool, default=True
        If True, center the data by subtracting the mean (standard PCA).
        If False, also scales the data (correlation matrix based PCA).

    whiten : bool, default=False
        If True, the transformation ensures uncorrelated outputs with unit variances 
        by dividing by the singular values.

    Attributes
    ----------
    components_ : np.ndarray of shape (n_components, n_features)
        Principal axes representing directions of maximum variance.

    explained_variance_ : np.ndarray of shape (n_components,)
        The amount of variance explained by each selected component.

    explained_variance_ratio_ : np.ndarray of shape (n_components,)
        Percentage of total variance explained by each component.

    singular_values_ : np.ndarray of shape (n_components,)
        Singular values (square roots of eigenvalues) from SVD.

    mean_ : np.ndarray of shape (n_features,)
        Per-feature empirical mean estimated from the training set.

    n_components_ : int
        Actual number of components kept.

    Notes
    -----
    **Complexity:**
    - :math:`O(min(n^2 p, n p^2))` where :math:`n` = samples, :math:`p` = features.

    **When to use:**
    - To reduce dimensionality while preserving global structure.
    - To visualize high-dimensional data (usually with 2 or 3 components).
    - To remove multicollinearity before regression or classification.
    - To compress data.

    **Limitations:**
    - Sensitive to the scale of the input features (scaling is recommended).
    - Only captures linear relationships.
    - Principal components are not always easily interpretable.

    References
    ----------
    .. [Jolliffe2002] Jolliffe, I. T. (2002). **Principal Component Analysis.** 
           *Springer Series in Statistics*, 2nd Edition.

    See Also
    --------
    :class:`~tuiml.features.extraction.RandomProjectionExtractor` : Faster, distance-preserving projection.

    Examples
    --------
    Reduce dimensions while explaining 95% of variance:

    >>> from tuiml.features.extraction import PCAExtractor
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> pca = PCAExtractor(n_components=0.95)
    >>> X_reduced = pca.fit_transform(X)
    >>> print(f"Reduced to {pca.n_components_} dimensions")
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        center: bool = True,
        whiten: bool = False
    ):
        super().__init__(n_components=n_components)
        self.n_components = n_components
        self.center = center
        self.whiten = whiten

        # Attributes set during fit
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PCAExtractor":
        """
        Fit the PCAExtractor model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : PCAExtractor
            The fitted PCAExtractor model.
        """
        X = self._ensure_numpy(X)
        n_samples, n_features = X.shape

        self._n_features_in = n_features
        self.n_features_in_ = n_features

        # Handle missing values by imputing with mean
        col_means = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X), col_means, X)

        # Center the data
        self.mean_ = np.mean(X, axis=0)

        if self.center:
            X_centered = X - self.mean_
            self.std_ = None
        else:
            # Standardize (correlation matrix)
            self.std_ = np.std(X, axis=0)
            self.std_[self.std_ == 0] = 1  # Avoid division by zero
            X_centered = (X - self.mean_) / self.std_

        # Compute SVD
        # X_centered = U * S * Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Eigenvalues from singular values
        # variance = eigenvalue = S^2 / (n_samples - 1)
        explained_variance = (S ** 2) / (n_samples - 1)
        total_variance = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_variance

        # Determine number of components
        if self.n_components is None:
            n_components = min(n_samples, n_features)
        elif isinstance(self.n_components, float):
            # Select components to explain this proportion of variance
            cumsum = np.cumsum(explained_variance_ratio)
            n_components = np.searchsorted(cumsum, self.n_components) + 1
            n_components = min(n_components, min(n_samples, n_features))
        else:
            n_components = min(self.n_components, min(n_samples, n_features))

        # Store results
        self.n_components_ = n_components
        self.components_ = Vt[:n_components]
        self.singular_values_ = S[:n_components]
        self.explained_variance_ = explained_variance[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:n_components]

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        self._check_is_fitted()
        X = self._ensure_numpy(X)

        # Handle missing values
        col_means = self.mean_
        X = np.where(np.isnan(X), col_means, X)

        # Center/standardize
        if self.center:
            X_centered = X - self.mean_
        else:
            X_centered = (X - self.mean_) / self.std_

        # Project onto principal components
        X_transformed = X_centered @ self.components_.T

        if self.whiten:
            X_transformed = X_transformed / self.singular_values_

        return X_transformed

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data back to its original space.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_components)
            Data in transformed space.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Data in original space.
        """
        self._check_is_fitted()
        X = self._ensure_numpy(X)

        if self.whiten:
            X = X * self.singular_values_

        # Project back to original space
        X_original = X @ self.components_

        # Reverse centering/standardization
        if self.center:
            X_original = X_original + self.mean_
        else:
            X_original = X_original * self.std_ + self.mean_

        return X_original

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the model and apply dimensionality reduction.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        return self.fit(X, y).transform(X)

    def get_covariance(self) -> np.ndarray:
        """
        Compute data covariance with the generative model.

        Returns
        -------
        cov : ndarray of shape (n_features, n_features)
            Estimated covariance of data.
        """
        self._check_is_fitted()

        components = self.components_
        exp_var = self.explained_variance_

        cov = (components.T * exp_var) @ components

        return cov

    def get_precision(self) -> np.ndarray:
        """
        Compute data precision matrix with the generative model.

        Returns
        -------
        precision : ndarray of shape (n_features, n_features)
            Estimated precision of data.
        """
        self._check_is_fitted()

        cov = self.get_covariance()
        precision = np.linalg.pinv(cov)

        return precision

    def _ensure_numpy(self, X) -> np.ndarray:
        """Convert input to NumPy array.

        Parameters
        ----------
        X : array-like
            Input data to convert.

        Returns
        -------
        result : np.ndarray
            Data as a NumPy array.
        """
        if hasattr(X, 'values'):
            return X.values
        return np.asarray(X)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : list of str, optional
            Ignored, output names are always PC1, PC2, etc.

        Returns
        -------
        feature_names_out : ndarray of str
            Names of the output features (PC1, PC2, ...).
        """
        self._check_is_fitted()
        return np.array([f"PC{i+1}" for i in range(self.n_components_)])

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_components": {
                "type": ["integer", "number", "null"],
                "default": None,
                "description": "Number of components to keep (int), variance proportion (float), or None for all"
            },
            "center": {
                "type": "boolean",
                "default": True,
                "description": "If True, center data (covariance). If False, standardize (correlation)."
            },
            "whiten": {
                "type": "boolean",
                "default": False,
                "description": "If True, whiten the components"
            }
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PCAExtractor(n_components={self.n_components}, "
            f"center={self.center}, whiten={self.whiten})"
        )
