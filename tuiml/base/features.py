"""
Base classes for feature engineering operations.

This module provides the foundation for feature selection, extraction,
and construction methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np

from tuiml.hub import registry, ComponentType, Registrable

class FeatureMethod(Registrable, ABC):
    """Abstract base class for all feature engineering operations.

    Defines the standard workflow for discovering, extracting, or creating 
    features from raw data.

    See Also
    --------
    :class:`~tuiml.base.features.FeatureSelector` : For subset selection.
    :class:`~tuiml.base.features.FeatureExtractor` : For dimensionality reduction.
    :class:`~tuiml.base.features.FeatureConstructor` : For expanding feature space.
    """

    def __init__(self):
        """Initialize feature method."""
        self._is_fitted = False
        self._feature_names_in: Optional[List[str]] = None
        self._feature_names_out: Optional[List[str]] = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FeatureMethod":
        """
        Learn from data.

        Args:
            X: Training data (n_samples, n_features)
            y: Target values (optional)

        Returns:
            Self (for method chaining)
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply transformation to data.

        Args:
            X: Data to transform

        Returns:
            Transformed data
        """
        pass

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Training data
            y: Target values (optional)

        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)

    def get_params(self) -> Dict[str, Any]:
        """Get method parameters."""
        params = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                params[key] = value
        return params

    def set_params(self, **params) -> "FeatureMethod":
        """Set method parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def _check_is_fitted(self):
        """Check if method has been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling transform"
            )

    def get_feature_names_out(self) -> Optional[List[str]]:
        """Return output feature names after transformation."""
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
    """

    _component_type = ComponentType.FEATURE_SELECTOR

    def __init__(self, k: Optional[int] = None, threshold: Optional[float] = None):
        """
        Initialize feature selector.

        Args:
            k: Number of features to select (if None, use threshold)
            threshold: Score threshold for selection (if None, use k)
        """
        super().__init__()
        self.k = k
        self.threshold = threshold
        self._selected_indices: Optional[np.ndarray] = None
        self._feature_scores: Optional[np.ndarray] = None

    @abstractmethod
    def _compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute feature scores.

        Args:
            X: Training data
            y: Target values

        Returns:
            Array of feature scores
        """
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FeatureSelector":
        """
        Compute feature scores and select features.

        Args:
            X: Training data
            y: Target values

        Returns:
            Self
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
        """
        Select features from data.

        Args:
            X: Data to transform

        Returns:
            Data with selected features only
        """
        self._check_is_fitted()
        return X[:, self._selected_indices]

    def get_selected_indices(self) -> np.ndarray:
        """Return indices of selected features."""
        self._check_is_fitted()
        return self._selected_indices

    def get_feature_scores(self) -> np.ndarray:
        """Return computed feature scores."""
        self._check_is_fitted()
        return self._feature_scores

    def get_support(self, indices: bool = False) -> np.ndarray:
        """
        Get a mask or indices of selected features.

        Args:
            indices: If True, return indices; if False, return boolean mask

        Returns:
            Mask or indices array
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
    """

    _component_type = ComponentType.FEATURE_EXTRACTOR

    def __init__(self, n_components: Optional[int] = None):
        """
        Initialize feature extractor.

        Args:
            n_components: Number of components to extract
        """
        super().__init__()
        self.n_components = n_components

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FeatureExtractor":
        """
        Learn extraction parameters from data.

        Args:
            X: Training data
            y: Target values (optional)

        Returns:
            Self
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract features from data.

        Args:
            X: Data to transform

        Returns:
            Extracted features
        """
        pass

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the transformation (if possible).

        Args:
            X: Transformed data

        Returns:
            Reconstructed data in original space

        Raises:
            NotImplementedError: If inverse is not supported
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse_transform"
        )

class FeatureConstructor(FeatureMethod):
    """Base class for expanded feature construction.

    Feature construction creates new features through combinations or 
    non-linear expansions of existing inputs (e.g., Polynomial Interactions, 
    Logarithmic mappings).
    """

    _component_type = ComponentType.FEATURE_CONSTRUCTOR

    def __init__(self):
        """Initialize feature constructor."""
        super().__init__()

    @abstractmethod
    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "FeatureConstructor":
        """
        Learn construction parameters from data.

        Args:
            X: Training data
            y: Target values (optional)

        Returns:
            Self
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Construct new features from data.

        Args:
            X: Data to transform

        Returns:
            Data with new features
        """
        pass

# Decorator shortcuts for registration
def feature_selector(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """
    Decorator to register a feature selector.

    Example::

        @feature_selector(tags=["statistical", "univariate"])
        class ChiSquaredSelector(FeatureSelector):
            pass
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
    """
    Decorator to register a feature extractor.

    Example::

        @feature_extractor(tags=["dimensionality_reduction"])
        class PCAExtractor(FeatureExtractor):
            pass
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
    """
    Decorator to register a feature constructor.

    Example::

        @feature_constructor(tags=["polynomial"])
        class PolynomialFeaturesGenerator(FeatureConstructor):
            pass
    """
    return registry.register(
        ComponentType.FEATURE_CONSTRUCTOR,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )
