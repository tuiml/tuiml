"""Scikit-Learn PCA wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional, Union

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    PCA = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["decomposition", "sklearn"], version="1.0.0")
class SklearnPCA(Transformer):
    """
    Principal component analysis (PCA) using Scikit-Learn.

    Linear dimensionality reduction using Singular Value Decomposition of the data
    to project it to a lower dimensional space.

    Parameters
    ----------
    n_components : int, float or 'mle', default=None
        Number of components to keep.
    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        If auto: The solver is selected by a default policy based on X.shape and n_components.
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float, str]] = None,
        svd_solver: str = "auto"
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_components = n_components
        self.svd_solver = svd_solver

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_components": {
                "type": ["integer", "number", "string", "null"],
                "default": None,
                "description": "Number of components to keep. string should be 'mle'"
            },
            "svd_solver": {
                "type": "string",
                "default": "auto",
                "enum": ["auto", "full", "arpack", "randomized"],
                "description": "The solver is selected by a default policy"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnPCA":
        """Fit the model with X."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = PCA(
            n_components=self.n_components,
            svd_solver=self.svd_solver
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction to X."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnPCA(n_components={self.n_components}, svd_solver='{self.svd_solver}')"
