"""Scikit-Learn Extra Trees wrappers."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, Regressor, classifier, regressor

try:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    ExtraTreesClassifier = ExtraTreesRegressor = None
    SKLEARN_AVAILABLE = False


@classifier(tags=["ensemble", "trees", "random-forest", "sklearn"], version="1.0.0")
class SklearnExtraTreesClassifier(Classifier):
    """Scikit-Learn Extra Trees Classifier."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, random_state: Optional[int] = None):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "n_estimators": {"type": "integer", "default": 100},
            "max_depth": {"type": ["integer", "null"], "default": None}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "nominal", "missing_values"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Algorithm":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.model_ = ExtraTreesClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)
        self.model_.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.predict_proba(np.asarray(X, dtype=float))


@regressor(tags=["ensemble", "trees", "random-forest", "sklearn"], version="1.0.0")
class SklearnExtraTreesRegressor(Regressor):
    """Scikit-Learn Extra Trees Regressor."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, random_state: Optional[int] = None):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "n_estimators": {"type": "integer", "default": 100},
            "max_depth": {"type": ["integer", "null"], "default": None}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "missing_values"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Algorithm":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.model_ = ExtraTreesRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)
        self.model_.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.predict(np.asarray(X, dtype=float))
