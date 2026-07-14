"""Scikit-Learn FactorAnalysis wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.decomposition import FactorAnalysis
    SKLEARN_AVAILABLE = True
except ImportError:
    FactorAnalysis = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["decomposition", "sklearn"], version="1.0.0")
class SklearnFactorAnalysis(Transformer):
    """
    Factor Analysis (FA) using Scikit-Learn.

    A simple linear generative model with Gaussian latent variables.

    Parameters
    ----------
    n_components : int, default=None
        Dimensionality of latent space, the number of components of X that are
        obtained after transform.
    tol : float, default=1e-2
        Stopping tolerance for EM algorithm.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        tol: float = 1e-2
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_components = n_components
        self.tol = tol

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_components": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Dimensionality of latent space"
            },
            "tol": {
                "type": "number",
                "default": 1e-2,
                "description": "Stopping tolerance for EM algorithm"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnFactorAnalysis":
        """Fit the FactorAnalysis model to X using EM."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = FactorAnalysis(
            n_components=self.n_components,
            tol=self.tol
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction to X using the model."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnFactorAnalysis(n_components={self.n_components})"
