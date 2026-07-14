"""Scikit-Learn MiniBatchNMF wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.decomposition import MiniBatchNMF
    SKLEARN_AVAILABLE = True
except ImportError:
    MiniBatchNMF = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["decomposition", "sklearn"], version="1.0.0")
class SklearnMiniBatchNMF(Transformer):
    """
    Mini-Batch Non-Negative Matrix Factorization (NMF) using Scikit-Learn.

    Parameters
    ----------
    n_components : int, default=None
        Number of components.
    init : str, default='warn'
        Method used to initialize the procedure.
    batch_size : int, default=1024
        Number of samples in each mini-batch.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        init: Optional[str] = None,
        batch_size: int = 1024
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_components = n_components
        self.init = init
        self.batch_size = batch_size

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
            "batch_size": {
                "type": "integer",
                "default": 1024,
                "description": "Number of samples in each mini-batch"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnMiniBatchNMF":
        """Learn a NMF model for the data X."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = MiniBatchNMF(
            n_components=self.n_components,
            init=self.init,
            batch_size=self.batch_size
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
        return f"SklearnMiniBatchNMF(n_components={self.n_components}, batch_size={self.batch_size})"
