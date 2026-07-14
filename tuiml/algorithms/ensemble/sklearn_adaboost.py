"""Scikit-Learn AdaBoost wrappers."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, Regressor, classifier, regressor

try:
    from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    AdaBoostClassifier = AdaBoostRegressor = None
    SKLEARN_AVAILABLE = False

@classifier(tags=["ensemble", "sklearn"], version="1.0.0")
class SklearnAdaBoostClassifier(Classifier):
    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0, random_state: Optional[int] = None):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model_ = None
        self.classes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "n_estimators": {"type": "integer", "default": 50},
            "learning_rate": {"type": "number", "default": 1.0}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "nominal", "multi_class"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnAdaBoostClassifier":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim == 1: X = X.reshape(-1, 1)
        self.model_ = AdaBoostClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate, random_state=self.random_state)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return self.model_.predict_proba(X)


@regressor(tags=["ensemble", "sklearn"], version="1.0.0")
class SklearnAdaBoostRegressor(Regressor):
    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0, random_state: Optional[int] = None):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {"n_estimators": {"type": "integer", "default": 50}, "learning_rate": {"type": "number", "default": 1.0}}

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "nominal"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnAdaBoostRegressor":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        self.model_ = AdaBoostRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate, random_state=self.random_state)
        self.model_.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return self.model_.predict(X)
