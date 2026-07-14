"""Scikit-Learn feature selection algorithms."""

import numpy as np
from typing import Dict, List, Any, Optional, Union

from tuiml.base.features import FeatureSelector, feature_selector
from tuiml.features.selection._base import SelectorMixin

try:
    from sklearn.feature_selection import (
        VarianceThreshold,
        SelectKBest,
        SelectPercentile,
        RFE,
        SelectFromModel,
        f_classif,
    )
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    VarianceThreshold = SelectKBest = SelectPercentile = RFE = SelectFromModel = f_classif = RandomForestClassifier = None
    SKLEARN_AVAILABLE = False


@feature_selector(tags=["selection", "variance", "sklearn"], version="1.0.0")
class SklearnVarianceThreshold(FeatureSelector, SelectorMixin):
    """Scikit-Learn VarianceThreshold."""

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.threshold = threshold
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"threshold": {"type": "number", "default": 0.0}}

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X = np.asarray(X, dtype=float)
        self._n_features_in = X.shape[1]
        self.model_ = VarianceThreshold(threshold=self.threshold)
        self.model_.fit(X)
        self._selected_indices = self.model_.get_support(indices=True)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.transform(np.asarray(X, dtype=float))


@feature_selector(tags=["selection", "univariate", "sklearn"], version="1.0.0")
class SklearnSelectKBest(FeatureSelector, SelectorMixin):
    """Scikit-Learn SelectKBest."""

    def __init__(self, k: Union[int, str] = 10):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.k = k
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"k": {"type": ["integer", "string"], "default": 10}}

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self._n_features_in = X.shape[1]
        self.model_ = SelectKBest(score_func=f_classif, k=self.k)
        self.model_.fit(X, y)
        self._selected_indices = self.model_.get_support(indices=True)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.transform(np.asarray(X, dtype=float))


@feature_selector(tags=["selection", "univariate", "sklearn"], version="1.0.0")
class SklearnSelectPercentile(FeatureSelector, SelectorMixin):
    """Scikit-Learn SelectPercentile."""

    def __init__(self, percentile: int = 10):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.percentile = percentile
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"percentile": {"type": "integer", "default": 10}}

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self._n_features_in = X.shape[1]
        self.model_ = SelectPercentile(score_func=f_classif, percentile=self.percentile)
        self.model_.fit(X, y)
        self._selected_indices = self.model_.get_support(indices=True)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.transform(np.asarray(X, dtype=float))


@feature_selector(tags=["selection", "wrapper", "rfe", "sklearn"], version="1.0.0")
class SklearnRFE(FeatureSelector, SelectorMixin):
    """Scikit-Learn Recursive Feature Elimination."""

    def __init__(self, n_features_to_select: Optional[Union[int, float]] = None, step: Union[int, float] = 1):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "n_features_to_select": {"type": ["integer", "number", "null"], "default": None},
            "step": {"type": ["integer", "number"], "default": 1}
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self._n_features_in = X.shape[1]
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model_ = RFE(estimator=estimator, n_features_to_select=self.n_features_to_select, step=self.step)
        self.model_.fit(X, y)
        self._selected_indices = self.model_.get_support(indices=True)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.transform(np.asarray(X, dtype=float))


@feature_selector(tags=["selection", "embedded", "sklearn"], version="1.0.0")
class SklearnSelectFromModel(FeatureSelector, SelectorMixin):
    """Scikit-Learn SelectFromModel."""

    def __init__(self, threshold: Optional[Union[str, float]] = None, max_features: Optional[int] = None):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.threshold = threshold
        self.max_features = max_features
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "threshold": {"type": ["string", "number", "null"], "default": None},
            "max_features": {"type": ["integer", "null"], "default": None}
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self._n_features_in = X.shape[1]
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        estimator.fit(X, y)
        self.model_ = SelectFromModel(estimator=estimator, threshold=self.threshold, max_features=self.max_features, prefit=True)
        self._selected_indices = self.model_.get_support(indices=True)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.transform(np.asarray(X, dtype=float))
