"""Scikit-Learn SparsePCA wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.decomposition import SparsePCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SparsePCA = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["decomposition", "sklearn"], version="1.0.0")
class SklearnSparsePCA(Transformer):
    """
    Sparse Principal Components Analysis (SparsePCA) using Scikit-Learn.

    Finds the set of sparse components that can optimally reconstruct the data.

    Parameters
    ----------
    n_components : int, default=None
        Number of sparse atoms to extract.
    alpha : float, default=1.0
        Sparsity controlling parameter. Higher values lead to sparser components.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        alpha: float = 1.0
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_components = n_components
        self.alpha = alpha

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_components": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Number of sparse atoms to extract"
            },
            "alpha": {
                "type": "number",
                "default": 1.0,
                "description": "Sparsity controlling parameter"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnSparsePCA":
        """Fit the model from data in X."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = SparsePCA(
            n_components=self.n_components,
            alpha=self.alpha
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Least Squares projection of the data onto the sparse components."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnSparsePCA(n_components={self.n_components}, alpha={self.alpha})"
