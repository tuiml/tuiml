"""Scikit-Learn FastICA wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.decomposition import FastICA
    SKLEARN_AVAILABLE = True
except ImportError:
    FastICA = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["decomposition", "sklearn"], version="1.0.0")
class SklearnFastICA(Transformer):
    """
    FastICA using Scikit-Learn.

    A fast algorithm for Independent Component Analysis.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to use.
    algorithm : {'parallel', 'deflation'}, default='parallel'
        Apply parallel or deflational algorithm for FastICA.
    whiten : str or bool, default='warn'
        If whiten is false, the data is already considered to be whitened,
        and no whitening is performed.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        algorithm: str = "parallel",
        whiten: str = "arbitrary-variance"
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_components": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Number of components to use"
            },
            "algorithm": {
                "type": "string",
                "default": "parallel",
                "enum": ["parallel", "deflation"],
                "description": "Apply parallel or deflational algorithm for FastICA"
            },
            "whiten": {
                "type": "string",
                "default": "arbitrary-variance",
                "description": "Whitening strategy"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnFastICA":
        """Fit the model to X."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = FastICA(
            n_components=self.n_components,
            algorithm=self.algorithm,
            whiten=self.whiten
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Recover the estimated sources from X."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the sources back to the mixed data."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.inverse_transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnFastICA(n_components={self.n_components})"
