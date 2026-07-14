"""Scikit-Learn Gaussian Process wrappers."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, Regressor, classifier, regressor

try:
    from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    GaussianProcessClassifier = GaussianProcessRegressor = None
    SKLEARN_AVAILABLE = False


@classifier(tags=["bayesian", "gaussian-process", "sklearn"], version="1.0.0")
class SklearnGaussianProcessClassifier(Classifier):
    """Scikit-Learn Gaussian Process Classifier."""

    def __init__(self, random_state: Optional[int] = None):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.random_state = random_state
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Algorithm":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.model_ = GaussianProcessClassifier(random_state=self.random_state)
        self.model_.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.predict_proba(np.asarray(X, dtype=float))


@regressor(tags=["bayesian", "gaussian-process", "sklearn"], version="1.0.0")
class SklearnGaussianProcessRegressor(Regressor):
    """Scikit-Learn Gaussian Process Regressor."""

    def __init__(self, alpha: float = 1e-10, random_state: Optional[int] = None):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.alpha = alpha
        self.random_state = random_state
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"alpha": {"type": "number", "default": 1e-10}}

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Algorithm":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.model_ = GaussianProcessRegressor(alpha=self.alpha, random_state=self.random_state)
        self.model_.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.predict(np.asarray(X, dtype=float))
