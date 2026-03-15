"""FilteredClassifier implementation."""

import numpy as np
from typing import Dict, List, Any, Optional, Type

from tuiml.base.algorithms import Classifier, classifier
from tuiml.hub import registry

@classifier(tags=["meta", "filter", "preprocessing"], version="1.0.0")
class FilteredClassifier(Classifier):
    """FilteredClassifier meta-classifier.

    Applies a filter (preprocessor) to the data before training, and then 
    applies the same transformation before prediction. This enables 
    seamless integration of data preprocessing and classification.

    Parameters
    ----------
    base_classifier : str or class, default='C45TreeClassifier'
        The base classifier to use.
    filter : str or object, optional
        The filter/preprocessor to use. If string, it is looked up in the registry.
    filter_params : dict, optional
        Parameters to initialize the filter with.

    Attributes
    ----------
    classifier_ : Classifier
        The fitted base classifier.
    filter_ : object
        The fitted filter/preprocessor.
    classes_ : np.ndarray
        The unique class labels.

    Examples
    --------
    >>> from tuiml.algorithms.ensemble import FilteredClassifier
    >>> clf = FilteredClassifier(base_classifier='C45TreeClassifier', filter='StandardScaler')
    >>> clf.fit(X_train, y_train)
    FilteredClassifier(...)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(self, base_classifier: Any = 'C45TreeClassifier',
                 filter: Any = None,
                 filter_params: Optional[Dict] = None):
        super().__init__()
        self.base_classifier = base_classifier
        self.filter = filter
        self.filter_params = filter_params or {}
        self.classifier_ = None
        self.filter_ = None
        self.classes_ = None
        self._base_class = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "base_classifier": {
                "type": "string", "default": "C45TreeClassifier",
                "description": "Base classifier name"
            },
            "filter": {
                "type": "string", "default": None,
                "description": "Filter/preprocessor name or None"
            },
            "filter_params": {
                "type": "object", "default": {},
                "description": "Parameters for the filter"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        return "O(filter_complexity + base_complexity)"

    @classmethod
    def get_references(cls) -> List[str]:
        return ["Meta-classifier that applies filtering before classification."]

    def _get_base_class(self) -> Type[Classifier]:
        """Get the base classifier class."""
        if isinstance(self.base_classifier, str):
            return registry.get(self.base_classifier)
        elif isinstance(self.base_classifier, type):
            return self.base_classifier
        elif isinstance(self.base_classifier, Classifier):
            # Handle classifier instances by extracting their class
            return type(self.base_classifier)
        else:
            raise ValueError(f"Invalid base_classifier: {self.base_classifier}")

    def _get_filter(self):
        """Get the filter/preprocessor."""
        if self.filter is None:
            return None

        if isinstance(self.filter, str):
            # Try to get from hub or use built-in preprocessors
            try:
                filter_class = registry.get(self.filter)
                return filter_class(**self.filter_params)
            except (KeyError, ValueError):
                # Use a simple standardization if filter not found
                return StandardScalerSimple()

        elif hasattr(self.filter, 'fit_transform'):
            # Already a filter instance
            return self.filter

        elif isinstance(self.filter, type):
            return self.filter(**self.filter_params)

        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FilteredClassifier":
        """Fit the FilteredClassifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : FilteredClassifier
            Returns the fitted instance.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.classes_ = np.unique(y)
        self._base_class = self._get_base_class()

        # Apply filter if specified
        self.filter_ = self._get_filter()
        if self.filter_ is not None:
            X_filtered = self.filter_.fit_transform(X)
        else:
            X_filtered = X

        # Fit base classifier
        self.classifier_ = self._base_class()
        self.classifier_.fit(X_filtered, y)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Apply filter if specified
        if self.filter_ is not None:
            X_filtered = self.filter_.transform(X)
        else:
            X_filtered = X

        return self.classifier_.predict(X_filtered)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Apply filter if specified
        if self.filter_ is not None:
            X_filtered = self.filter_.transform(X)
        else:
            X_filtered = X

        return self.classifier_.predict_proba(X_filtered)

    def __repr__(self) -> str:
        if self._is_fitted:
            return f"FilteredClassifier(base={self.base_classifier}, filter={self.filter})"
        return f"FilteredClassifier(base_classifier={self.base_classifier})"

class StandardScalerSimple:
    """Simple standardization (z-score normalization)."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "StandardScalerSimple":
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ == 0] = 1.0  # Avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
