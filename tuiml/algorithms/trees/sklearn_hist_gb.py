"""Scikit-Learn Histogram-based Gradient Boosting wrappers."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, Regressor, classifier, regressor

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    HistGradientBoostingClassifier = None
    HistGradientBoostingRegressor = None
    SKLEARN_AVAILABLE = False


@classifier(tags=["ensemble", "trees", "sklearn", "fast"], version="1.0.0")
class SklearnHistGradientBoostingClassifier(Classifier):
    """
    Histogram-based Gradient Boosting Classification Tree.
    
    This estimator is much faster than GradientBoostingClassifier for big datasets (n_samples >= 10 000).
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: int = 31,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        super().__init__()
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed.")
            
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.random_state = random_state

        self.model_ = None
        self.classes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "learning_rate": {"type": "number", "default": 0.1},
            "max_iter": {"type": "integer", "default": 100},
            "max_leaf_nodes": {"type": "integer", "default": 31},
            "max_depth": {"type": ["integer", "null"], "default": None}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "nominal", "missing_values", "multi_class"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnHistGradientBoostingClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1: X = X.reshape(-1, 1)

        self.model_ = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
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


@regressor(tags=["ensemble", "trees", "sklearn", "fast"], version="1.0.0")
class SklearnHistGradientBoostingRegressor(Regressor):
    """
    Histogram-based Gradient Boosting Regression Tree.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: int = 31,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        super().__init__()
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed.")
            
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.random_state = random_state
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "learning_rate": {"type": "number", "default": 0.1},
            "max_iter": {"type": "integer", "default": 100},
            "max_leaf_nodes": {"type": "integer", "default": 31},
            "max_depth": {"type": ["integer", "null"], "default": None}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        return ["numeric", "missing_values"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnHistGradientBoostingRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)

        self.model_ = HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model_.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return self.model_.predict(X)
