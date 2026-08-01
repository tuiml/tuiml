"""
Base classes for preprocessing operations.

This module provides the foundation for all data preprocessing transformers.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import numpy as np

from tuiml.registry import registry, ComponentType, Registrable

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

    See Also
    --------
    :class:`~tuiml.base.preprocessing.Filter` : For value removal/replacement.
    :class:`~tuiml.base.preprocessing.Transformer` : For feature-space
        transformations.
    :class:`~tuiml.base.preprocessing.InstanceTransformer` : For row-level
        resampling.
    """

    _component_type = ComponentType.PREPROCESSOR

    def __init__(self):
        """Initialize preprocessor state."""
        self._is_fitted = False

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Preprocessor":
        """Learn preprocessing parameters from data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,), optional
            Target values. Required only by supervised preprocessors.

        Returns
        -------
        self : Preprocessor
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
            Target values. Required only by supervised preprocessors.

        Returns
        -------
        X_transformed : np.ndarray
            Transformed data.
        """
        return self.fit(X, y).transform(X)

    def get_params(self) -> dict:
        """Return the preprocessor's public parameters.

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

    def set_params(self, **params) -> "Preprocessor":
        """Set transformation parameters.

        Parameters
        ----------
        **params : dict
            Parameter names mapped to new values. Each name must match an
            existing attribute.

        Returns
        -------
        self : Preprocessor
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
        """Raise ``RuntimeError`` if the preprocessor has not been fitted yet."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling transform"
            )

class Filter(Preprocessor):
    """Base class for filter-type preprocessors.

    Filters typically modify the data by removing or replacing values
    without changing the underlying math of the feature space (e.g.,
    handling missing values, removing outliers).

    See Also
    --------
    :class:`~tuiml.base.preprocessing.Transformer` : For feature-space
        transformations.
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

    See Also
    --------
    :class:`~tuiml.base.preprocessing.SupervisedTransformer` : Transformers
        that use target labels during fitting.
    :class:`~tuiml.base.preprocessing.Filter` : For value removal/replacement.
    """

    _component_type = ComponentType.TRANSFORMER
    _n_features_in: int = None
    _feature_names_in: List[str] = None

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Convert input to a 2D float numpy array."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the transformation (if possible).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Transformed data.

        Returns
        -------
        X_original : np.ndarray
            Data mapped back to the original scale/space.

        Raises
        ------
        NotImplementedError
            If the transformer does not support inversion.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse_transform"
        )

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names.

        Parameters
        ----------
        input_features : list of str, optional
            Input feature names. If given, they are returned as-is;
            otherwise names seen during :meth:`fit` (or generated
            ``x0, x1, ...`` placeholders) are used.

        Returns
        -------
        feature_names_out : list of str
            Output feature names.
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

    See Also
    --------
    :class:`~tuiml.base.preprocessing.Transformer` : Unsupervised feature
        transformers.
    """

    _supervised = True

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupervisedTransformer":
        """Learn transformation parameters from data and targets.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.
        y : np.ndarray of shape (n_samples,)
            Target values (required).

        Returns
        -------
        self : SupervisedTransformer
            The fitted instance (for method chaining).
        """
        pass

class InstanceTransformer(Preprocessor):
    """Base class for instance-level transformations.

    Specialized preprocessors that can change the row count of a dataset,
    such as resampling algorithms or extreme outlier removers.

    See Also
    --------
    :class:`~tuiml.base.preprocessing.Filter` : Column-preserving filters.
    """

    _component_type = ComponentType.FILTER

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> tuple:
        """Transform instances, possibly changing the number of rows.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.
        y : np.ndarray of shape (n_samples,), optional
            Target values.

        Returns
        -------
        X_transformed : np.ndarray
            Transformed data (row count may differ from the input).
        y_transformed : np.ndarray or None
            Correspondingly transformed target values.
        """
        pass

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> tuple:
        """Fit to data, then transform it in one step.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.
        y : np.ndarray of shape (n_samples,), optional
            Target values.

        Returns
        -------
        X_transformed : np.ndarray
            Transformed data (row count may differ from the input).
        y_transformed : np.ndarray or None
            Correspondingly transformed target values.
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
    """Register a preprocessor class with the component registry.

    Parameters
    ----------
    name : str, optional
        Registration name. Defaults to the class name.
    tags : list of str, optional
        Searchable tags describing the preprocessor.
    version : str, default="1.0.0"
        Component version string.
    author : str, optional
        Component author.

    Returns
    -------
    decorator : callable
        Class decorator that registers the decorated class in the registry
        as a :class:`~tuiml.base.preprocessing.Preprocessor` component.

    Examples
    --------
    >>> from tuiml.base.preprocessing import preprocessor, Transformer
    >>> @preprocessor(tags=["normalization"])
    ... class MinMaxScaler(Transformer):
    ...     pass
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
    """Register a filter class with the component registry.

    Parameters
    ----------
    name : str, optional
        Registration name. Defaults to the class name.
    tags : list of str, optional
        Searchable tags describing the filter.
    version : str, default="1.0.0"
        Component version string.
    author : str, optional
        Component author.

    Returns
    -------
    decorator : callable
        Class decorator that registers the decorated class in the registry
        as a :class:`~tuiml.base.preprocessing.Filter` component.

    Examples
    --------
    >>> from tuiml.base.preprocessing import filter_method, Filter
    >>> @filter_method(tags=["missing_values"])
    ... class MissingValueHandler(Filter):
    ...     pass
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
    """Register a transformer class with the component registry.

    Parameters
    ----------
    name : str, optional
        Registration name. Defaults to the class name.
    tags : list of str, optional
        Searchable tags describing the transformer.
    version : str, default="1.0.0"
        Component version string.
    author : str, optional
        Component author.

    Returns
    -------
    decorator : callable
        Class decorator that registers the decorated class in the registry
        as a :class:`~tuiml.base.preprocessing.Transformer` component.

    Examples
    --------
    >>> from tuiml.base.preprocessing import transformer, Transformer
    >>> @transformer(tags=["scaling", "normalization"])
    ... class StandardScaler(Transformer):
    ...     pass
    """
    return registry.register(
        ComponentType.TRANSFORMER,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )
