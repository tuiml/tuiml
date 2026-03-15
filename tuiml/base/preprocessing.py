"""
Base classes for preprocessing operations.

This module provides the foundation for all data preprocessing transformers.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import numpy as np

from tuiml.hub import registry, ComponentType, Registrable

class Preprocessor(Registrable, ABC):
    """Abstract base class for all data preprocessing operations.

    Defines a consistent API for fitting parameters to training data and 
    applying those parameters to transform data.

    Overview
    --------
    Preprocessors are the building blocks of data pipelines. They can be 
    unsupervised (only X) or supervised (X and y).

    Notes
    -----
    Subclasses must implement :meth:`fit` and :meth:`transform`.
    """

    _component_type = ComponentType.PREPROCESSOR

    def __init__(self):
        """Initialize preprocessor."""
        self._is_fitted = False

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for preprocessor parameters."""
        return {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Preprocessor":
        """
        Learn parameters from data.

        Args:
            X: Training data (n_samples, n_features)
            y: Target values (optional, n_samples,)

        Returns:
            Self (for method chaining)
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply transformation to data.

        Args:
            X: Data to transform (n_samples, n_features)

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

    def get_params(self) -> dict:
        """
        Get transformation parameters.

        Returns:
            Dictionary of parameters
        """
        params = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                params[key] = value
        return params

    def set_params(self, **params) -> "Preprocessor":
        """
        Set transformation parameters.

        Args:
            **params: Parameters to set

        Returns:
            Self (for method chaining)
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def _check_is_fitted(self):
        """Check if preprocessor has been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling transform"
            )

class Filter(Preprocessor):
    """Base class for filter-type preprocessors.

    Filters typically modify the data by removing or replacing values 
    without changing the underlying math of the feature space (e.g., 
    handling missing values, removing outliers).
    """

    _component_type = ComponentType.FILTER

class Transformer(Preprocessor):
    """Base class for feature transformers.

    Transformers apply mathematical operations to change the scale, 
    distribution, or representation of feature values.

    Attributes
    ----------
    _n_features_in : int
        Number of input features expected by the transformer.
    _feature_names_in : list of str
        The names of the features seen during :meth:`fit`.
    """

    _component_type = ComponentType.TRANSFORMER
    _n_features_in: int = None
    _feature_names_in: List[str] = None

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and convert input to numpy array."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the transformation (if possible).

        Args:
            X: Transformed data

        Returns:
            Data in original scale/space

        Raises:
            NotImplementedError: If inverse is not supported
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse_transform"
        )

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names.

        Args:
            input_features: Input feature names

        Returns:
            Output feature names
        """
        self._check_is_fitted()
        if input_features is not None:
            return list(input_features)
        if self._feature_names_in is not None:
            return list(self._feature_names_in)
        return [f"x{i}" for i in range(self._n_features_in)]

class SupervisedTransformer(Transformer):
    """Base class for supervised feature transformers.

    Unlike standard transformers, supervised transformers utilize the target 
    labels (:math:`y`) during the :meth:`fit` process to optimize the 
    transformation (e.g., Target Encoding, Decision-Tree Binning).
    """

    _supervised = True

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupervisedTransformer":
        """
        Fit the transformer to data.

        Args:
            X: Input data (n_samples, n_features)
            y: Target values (required)

        Returns:
            Self for method chaining
        """
        pass

class InstanceTransformer(Preprocessor):
    """Base class for instance-level transformations.

    Specialized preprocessors that can change the row count of a dataset, 
    such as resampling algorithms or extreme outlier removers.
    """

    _component_type = ComponentType.FILTER

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> tuple:
        """
        Transform instances.

        Args:
            X: Input data (n_samples, n_features)
            y: Target values (optional)

        Returns:
            (X_transformed, y_transformed) tuple
        """
        pass

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> tuple:
        """
        Fit and transform in one step.

        Args:
            X: Input data
            y: Target values (optional)

        Returns:
            (X_transformed, y_transformed) tuple
        """
        self.fit(X, y)
        return self.transform(X, y)

# Decorator shortcuts for registration
def preprocessor(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """
    Decorator to register a preprocessor.

    Example:
        @preprocessor(tags=["normalization"])
        class MinMaxScaler(Transformer):
            pass
    """
    return registry.register(
        ComponentType.PREPROCESSOR,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )

def filter_method(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """
    Decorator to register a filter.

    Example:
        @filter_method(tags=["missing_values"])
        class MissingValueHandler(Filter):
            pass
    """
    return registry.register(
        ComponentType.FILTER,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )

def transformer(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """
    Decorator to register a transformer.

    Example:
        @transformer(tags=["scaling", "normalization"])
        class StandardScaler(Transformer):
            pass
    """
    return registry.register(
        ComponentType.TRANSFORMER,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )
