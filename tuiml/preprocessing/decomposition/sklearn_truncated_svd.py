"""Scikit-Learn TruncatedSVD wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    TruncatedSVD = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["decomposition", "sklearn"], version="1.0.0")
class SklearnTruncatedSVD(Transformer):
    """
    Dimensionality reduction using truncated SVD (aka LSA) using Scikit-Learn.

    This transformer performs linear dimensionality reduction by means of
    truncated singular value decomposition (SVD).

    Parameters
    ----------
    n_components : int, default=2
        Desired dimensionality of output data.
    algorithm : {'arpack', 'randomized'}, default='randomized'
        SVD solver to use.
    n_iter : int, default=5
        Number of iterations for randomized SVD solver.
    """

    def __init__(
        self,
        n_components: int = 2,
        algorithm: str = "randomized",
        n_iter: int = 5
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_components = n_components
        self.algorithm = algorithm
        self.n_iter = n_iter

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_components": {
                "type": "integer",
                "default": 2,
                "description": "Desired dimensionality of output data"
            },
            "algorithm": {
                "type": "string",
                "default": "randomized",
                "enum": ["arpack", "randomized"],
                "description": "SVD solver to use"
            },
            "n_iter": {
                "type": "integer",
                "default": 5,
                "description": "Number of iterations for randomized SVD solver"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnTruncatedSVD":
        """Fit model on training data X."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = TruncatedSVD(
            n_components=self.n_components,
            algorithm=self.algorithm,
            n_iter=self.n_iter
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Perform dimensionality reduction on X."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X back to its original space."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.inverse_transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnTruncatedSVD(n_components={self.n_components}, algorithm='{self.algorithm}')"
