"""Scikit-Learn IncrementalPCA wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.decomposition import IncrementalPCA
    SKLEARN_AVAILABLE = True
except ImportError:
    IncrementalPCA = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["decomposition", "sklearn"], version="1.0.0")
class SklearnIncrementalPCA(Transformer):
    """
    Incremental principal components analysis (IPCA) using Scikit-Learn.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep.
    batch_size : int, default=None
        The number of samples to use for each batch.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        batch_size: Optional[int] = None
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_components = n_components
        self.batch_size = batch_size

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_components": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Number of components to keep"
            },
            "batch_size": {
                "type": ["integer", "null"],
                "default": None,
                "description": "The number of samples to use for each batch"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnIncrementalPCA":
        """Fit the model with X, using minibatches of size batch_size."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = IncrementalPCA(
            n_components=self.n_components,
            batch_size=self.batch_size
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
        return f"SklearnIncrementalPCA(n_components={self.n_components})"
