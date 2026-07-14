"""Scikit-Learn IterativeImputer wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    IterativeImputer = None
    SKLEARN_AVAILABLE = False


@transformer(tags=["imputation", "sklearn", "multivariate"], version="1.0.0")
class SklearnIterativeImputer(Transformer):
    """
    Multivariate imputer that estimates each feature from all the others.
    """

    def __init__(self, max_iter: int = 10, random_state: Optional[int] = None):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {"max_iter": {"type": "integer", "default": 10}}

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "missing_values"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnIterativeImputer":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        self.model_ = IterativeImputer(max_iter=self.max_iter, random_state=self.random_state)
        self.model_.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        return self.model_.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)
