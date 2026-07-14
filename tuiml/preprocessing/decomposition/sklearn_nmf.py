"""Scikit-Learn NMF wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.decomposition import NMF
    SKLEARN_AVAILABLE = True
except ImportError:
    NMF = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["decomposition", "sklearn"], version="1.0.0")
class SklearnNMF(Transformer):
    """
    Non-Negative Matrix Factorization (NMF) using Scikit-Learn.

    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X.

    Parameters
    ----------
    n_components : int, default=None
        Number of components, if n_components is not set all features
        are kept.
    init : str, default='warn'
        Method used to initialize the procedure.
    max_iter : int, default=200
        Maximum number of iterations before timing out.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        init: Optional[str] = None,
        max_iter: int = 200
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_components": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Number of components"
            },
            "init": {
                "type": ["string", "null"],
                "default": None,
                "description": "Method used to initialize the procedure"
            },
            "max_iter": {
                "type": "integer",
                "default": 200,
                "description": "Maximum number of iterations before timing out"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnNMF":
        """Learn a NMF model for the data X."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = NMF(
            n_components=self.n_components,
            init=self.init,
            max_iter=self.max_iter
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data X according to the fitted NMF model."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data back to its original space."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.inverse_transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnNMF(n_components={self.n_components})"
