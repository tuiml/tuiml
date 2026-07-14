"""Scikit-Learn MiniBatchSparsePCA wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.decomposition import MiniBatchSparsePCA
    SKLEARN_AVAILABLE = True
except ImportError:
    MiniBatchSparsePCA = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["decomposition", "sklearn"], version="1.0.0")
class SklearnMiniBatchSparsePCA(Transformer):
    """
    Mini-batch Sparse Principal Components Analysis using Scikit-Learn.

    Finds the set of sparse components that can optimally reconstruct the data.
    The amount of sparseness is controllable by the coefficient of the L1 penalty,
    given by the parameter alpha.

    Parameters
    ----------
    n_components : int, default=None
        Number of sparse atoms to extract.
    alpha : int, default=1
        Sparsity controlling parameter. Higher values lead to sparser components.
    batch_size : int, default=3
        The number of features to take in each mini batch.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        alpha: int = 1,
        batch_size: int = 3
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_components = n_components
        self.alpha = alpha
        self.batch_size = batch_size

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
                "type": "integer",
                "default": 1,
                "description": "Sparsity controlling parameter"
            },
            "batch_size": {
                "type": "integer",
                "default": 3,
                "description": "The number of features to take in each mini batch"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnMiniBatchSparsePCA":
        """Fit the model from data in X."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = MiniBatchSparsePCA(
            n_components=self.n_components,
            alpha=self.alpha,
            batch_size=self.batch_size
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
        return f"SklearnMiniBatchSparsePCA(n_components={self.n_components}, alpha={self.alpha})"
