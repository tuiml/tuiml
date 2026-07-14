"""Scikit-Learn Regularized Linear wrappers."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, Regressor, classifier, regressor

try:
    from sklearn.linear_model import Ridge, RidgeClassifier, Lasso, ElasticNet
    SKLEARN_AVAILABLE = True
except ImportError:
    Ridge = RidgeClassifier = Lasso = ElasticNet = None
    SKLEARN_AVAILABLE = False


@classifier(tags=["linear", "sklearn", "regularization"], version="1.0.0")
class SklearnRidgeClassifier(Classifier):
    def __init__(self, alpha: float = 1.0, max_iter: Optional[int] = None, random_state: Optional[int] = None):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_ = None
        self.classes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "alpha": {"type": "number", "default": 1.0},
            "max_iter": {"type": ["integer", "null"], "default": None}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "nominal", "multi_class"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnRidgeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1: X = X.reshape(-1, 1)
        self.model_ = RidgeClassifier(alpha=self.alpha, max_iter=self.max_iter, random_state=self.random_state)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return self.model_.predict(X)

@regressor(tags=["linear", "sklearn", "regularization"], version="1.0.0")
class SklearnRidge(Regressor):
    def __init__(self, alpha: float = 1.0, max_iter: Optional[int] = None, random_state: Optional[int] = None):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {"alpha": {"type": "number", "default": 1.0}, "max_iter": {"type": ["integer", "null"], "default": None}}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "nominal"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnRidge":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        self.model_ = Ridge(alpha=self.alpha, max_iter=self.max_iter, random_state=self.random_state)
        self.model_.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return self.model_.predict(X)

@regressor(tags=["linear", "sklearn", "regularization", "sparsity"], version="1.0.0")
class SklearnLasso(Regressor):
    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, random_state: Optional[int] = None):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {"alpha": {"type": "number", "default": 1.0}, "max_iter": {"type": "integer", "default": 1000}}

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "nominal"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnLasso":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        self.model_ = Lasso(alpha=self.alpha, max_iter=self.max_iter, random_state=self.random_state)
        self.model_.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return self.model_.predict(X)

@regressor(tags=["linear", "sklearn", "regularization", "sparsity"], version="1.0.0")
class SklearnElasticNet(Regressor):
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 1000, random_state: Optional[int] = None):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "alpha": {"type": "number", "default": 1.0},
            "l1_ratio": {"type": "number", "default": 0.5},
            "max_iter": {"type": "integer", "default": 1000}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "nominal"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnElasticNet":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        self.model_ = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter, random_state=self.random_state)
        self.model_.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return self.model_.predict(X)
