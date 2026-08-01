"""
Base classes for feature engineering operations.

This module provides the foundation for feature selection, extraction,
and construction methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np

from tuiml.registry import registry, ComponentType, Registrable

class FeatureMethod(Registrable, ABC):
    """Abstract base class for all feature engineering operations.

    Defines the standard workflow for discovering, extracting, or creating
    features from raw data. Concrete subclasses are registered with the
    component registry so they can be discovered by name.

    See Also
    --------
    :class:`~tuiml.base.features.FeatureSelector` : For subset selection.
    :class:`~tuiml.base.features.FeatureExtractor` : For dimensionality reduction.
    :class:`~tuiml.base.features.FeatureConstructor` : For expanding feature space.
    """

    def __init__(self):
        """Initialize feature method state."""
        self._is_fitted = False
        self._feature_names_in: Optional[List[str]] = None
        self._feature_names_out: Optional[List[str]] = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FeatureMethod":
        """Learn feature engineering parameters from data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,), optional
            Target values. Required only by supervised methods.

        Returns
        -------
        self : FeatureMethod
            The fitted instance (for method chaining).
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned transformation to data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : np.ndarray
            Transformed data.
        """
        pass

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit to data, then transform it in one step.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,), optional
            Target values. Required only by supervised methods.

        Returns
        -------
        X_transformed : np.ndarray
            Transformed data.
        """
        return self.fit(X, y).transform(X)

    def get_params(self) -> Dict[str, Any]:
        """Return the method's public parameters.

        Returns
        -------
        params : dict
            Mapping of parameter names to their current values.
        """
        params = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                params[key] = value
        return params

    def set_params(self, **params) -> "FeatureMethod":
        """Set method parameters.

        Parameters
        ----------
        **params : dict
            Parameter names mapped to new values. Each name must match an
            existing attribute.

        Returns
        -------
        self : FeatureMethod
            The updated instance (for method chaining).

        Raises
        ------
        ValueError
            If a parameter name does not match an existing attribute.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def _check_is_fitted(self):
        """Raise ``RuntimeError`` if the method has not been fitted yet."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling transform"
            )

    def get_feature_names_out(self) -> Optional[List[str]]:
        """Return output feature names after transformation.

        Returns
        -------
        feature_names_out : list of str or None
            Names of the output features, or ``None`` if unavailable.
        """
        return self._feature_names_out

class FeatureSelector(FeatureMethod):
    """Base class for feature selection algorithms.

    Feature selection identifies and preserves the most relevant subset of
    existing features based on statistical significance, model importance,
    or information theory.

    Overview
    --------
    Unlike extraction, selection does not create new features; it simplifies
    the model by pruning irrelevant or redundant inputs.

    Parameters
    ----------
    k : int, optional
        The number of top-scoring features to retain.
    threshold : float, optional
        The minimum score required for a feature to be selected.

    Attributes
    ----------
    _selected_indices : np.ndarray
        The indices of the features chosen during :meth:`fit`.
    _feature_scores : np.ndarray
        The raw scores calculated for each input feature.

    See Also
    --------
    :class:`~tuiml.base.features.FeatureExtractor` : For dimensionality reduction.
    :class:`~tuiml.base.features.FeatureConstructor` : For expanding feature space.
    """

    _component_type = ComponentType.FEATURE_SELECTOR

    def __init__(self, k: Optional[int] = None, threshold: Optional[float] = None):
        """Initialize feature selector.

        Parameters
        ----------
        k : int, optional
            Number of features to select. If ``None``, ``threshold`` is used.
        threshold : float, optional
            Score threshold for selection. If ``None``, ``k`` is used.
        """
        super().__init__()
        self.k = k
        self.threshold = threshold
        self._selected_indices: Optional[np.ndarray] = None
        self._feature_scores: Optional[np.ndarray] = None

    @abstractmethod
    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute a relevance score for each feature.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        scores : np.ndarray of shape (n_features,)
            Score for each feature (higher means more relevant).
        """
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FeatureSelector":
        """Compute feature scores and select the feature subset.

        Selection keeps the top ``k`` features by score, or all features whose
        score is at least ``threshold``. If neither is set, the top 10 features
        (or all, if fewer) are kept.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target values. Required for feature selection.

        Returns
        -------
        self : FeatureSelector
            The fitted instance (for method chaining).

        Raises
        ------
        ValueError
            If ``y`` is ``None``.
        """
        if y is None:
            raise ValueError("Feature selection requires target values (y)")

        self._feature_scores = self._compute_scores(X, y)

        # Select features based on k or threshold
        if self.k is not None:
            k = min(self.k, X.shape[1])
            self._selected_indices = np.argsort(self._feature_scores)[-k:]
        elif self.threshold is not None:
            self._selected_indices = np.where(
                self._feature_scores >= self.threshold
            )[0]
        else:
            # Default: select top 10 or all if fewer
            k = min(10, X.shape[1])
            self._selected_indices = np.argsort(self._feature_scores)[-k:]

        self._selected_indices = np.sort(self._selected_indices)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduce data to the selected features.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_selected)
            Data restricted to the selected feature columns.
        """
        self._check_is_fitted()
        return X[:, self._selected_indices]

    def get_selected_indices(self) -> np.ndarray:
        """Return indices of the selected features.

        Returns
        -------
        indices : np.ndarray of shape (n_selected,)
            Sorted indices of the features chosen during :meth:`fit`.
        """
        self._check_is_fitted()
        return self._selected_indices

    def get_feature_scores(self) -> np.ndarray:
        """Return the computed feature scores.

        Returns
        -------
        scores : np.ndarray of shape (n_features,)
            Score for each input feature.
        """
        self._check_is_fitted()
        return self._feature_scores

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get a mask or indices of the selected features.

        Parameters
        ----------
        indices : bool, default=False
            If ``True``, return the integer indices of the selected
            features; otherwise return a boolean mask over all features.

        Returns
        -------
        support : np.ndarray
            Boolean mask of shape ``(n_features,)``, or integer indices of
            shape ``(n_selected,)`` if ``indices`` is ``True``.
        """
        self._check_is_fitted()
        if indices:
            return self._selected_indices
        else:
            mask = np.zeros(len(self._feature_scores), dtype=bool)
            mask[self._selected_indices] = True
            return mask

class FeatureExtractor(FeatureMethod):
    """Base class for feature extraction and dimensionality reduction.

    Feature extraction transforms the original high-dimensional data into
    a lower-dimensional representation while preserving as much information
    as possible (e.g., PCA, SVD).

    Parameters
    ----------
    n_components : int, optional
        The number of projection components or latent dimensions to extract.

    See Also
    --------
    :class:`~tuiml.base.features.FeatureSelector` : For subset selection.
    :class:`~tuiml.base.features.FeatureConstructor` : For expanding feature space.
    """

    _component_type = ComponentType.FEATURE_EXTRACTOR

    def __init__(self, n_components: Optional[int] = None):
        """Initialize feature extractor.

        Parameters
        ----------
        n_components : int, optional
            Number of components to extract.
        """
        super().__init__()
        self.n_components = n_components

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FeatureExtractor":
        """Learn extraction parameters from data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,), optional
            Target values. Used only by supervised extractors.

        Returns
        -------
        self : FeatureExtractor
            The fitted instance (for method chaining).
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract features from data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : np.ndarray of shape (n_samples, n_components)
            Extracted features.
        """
        pass

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the transformation (if possible).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_components)
            Transformed data.

        Returns
        -------
        X_original : np.ndarray of shape (n_samples, n_features)
            Reconstructed data in the original feature space.

        Raises
        ------
        NotImplementedError
            If the extractor does not support inversion.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse_transform"
        )

class FeatureConstructor(FeatureMethod):
    """Base class for expanded feature construction.

    Feature construction creates new features through combinations or
    non-linear expansions of existing inputs (e.g., Polynomial Interactions,
    Logarithmic mappings).

    See Also
    --------
    :class:`~tuiml.base.features.FeatureSelector` : For subset selection.
    :class:`~tuiml.base.features.FeatureExtractor` : For dimensionality reduction.
    """

    _component_type = ComponentType.FEATURE_CONSTRUCTOR

    def __init__(self):
        """Initialize feature constructor."""
        super().__init__()

    @abstractmethod
    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "FeatureConstructor":
        """Learn construction parameters from data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,), optional
            Target values. Used only by supervised constructors.

        Returns
        -------
        self : FeatureConstructor
            The fitted instance (for method chaining).
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Construct new features from data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : np.ndarray
            Data augmented with the constructed features.
        """
        pass

# Decorator shortcuts for registration
def feature_selector(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """Register a feature selector class with the component registry.

    Parameters
    ----------
    name : str, optional
        Registration name. Defaults to the class name.
    tags : list of str, optional
        Searchable tags describing the selector.
    version : str, default="1.0.0"
        Component version string.
    author : str, optional
        Component author.

    Returns
    -------
    decorator : callable
        Class decorator that registers the decorated class in the registry
        as a :class:`~tuiml.base.features.FeatureSelector` component.

    Examples
    --------
    >>> from tuiml.base.features import feature_selector, FeatureSelector
    >>> @feature_selector(tags=["statistical", "univariate"])
    ... class ChiSquaredSelector(FeatureSelector):
    ...     pass
    """
    return registry.register(
        ComponentType.FEATURE_SELECTOR,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )

def feature_extractor(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """Register a feature extractor class with the component registry.

    Parameters
    ----------
    name : str, optional
        Registration name. Defaults to the class name.
    tags : list of str, optional
        Searchable tags describing the extractor.
    version : str, default="1.0.0"
        Component version string.
    author : str, optional
        Component author.

    Returns
    -------
    decorator : callable
        Class decorator that registers the decorated class in the registry
        as a :class:`~tuiml.base.features.FeatureExtractor` component.

    Examples
    --------
    >>> from tuiml.base.features import feature_extractor, FeatureExtractor
    >>> @feature_extractor(tags=["dimensionality_reduction"])
    ... class PCAExtractor(FeatureExtractor):
    ...     pass
    """
    return registry.register(
        ComponentType.FEATURE_EXTRACTOR,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )

def feature_constructor(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """Register a feature constructor class with the component registry.

    Parameters
    ----------
    name : str, optional
        Registration name. Defaults to the class name.
    tags : list of str, optional
        Searchable tags describing the constructor.
    version : str, default="1.0.0"
        Component version string.
    author : str, optional
        Component author.

    Returns
    -------
    decorator : callable
        Class decorator that registers the decorated class in the registry
        as a :class:`~tuiml.base.features.FeatureConstructor` component.

    Examples
    --------
    >>> from tuiml.base.features import feature_constructor, FeatureConstructor
    >>> @feature_constructor(tags=["polynomial"])
    ... class PolynomialFeaturesGenerator(FeatureConstructor):
    ...     pass
    """
    return registry.register(
        ComponentType.FEATURE_CONSTRUCTOR,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )
