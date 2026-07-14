"""Scikit-Learn KernelPCA wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.decomposition import KernelPCA
    SKLEARN_AVAILABLE = True
except ImportError:
    KernelPCA = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["decomposition", "sklearn"], version="1.0.0")
class SklearnKernelPCA(Transformer):
    """
    Kernel Principal component analysis (KPCA) using Scikit-Learn.

    Non-linear dimensionality reduction through the use of kernels.

    Parameters
    ----------
    n_components : int, default=None
        Number of components. If None, all non-zero components are kept.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'}, default='linear'
        Kernel used for PCA.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        kernel: str = "linear"
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_components = n_components
        self.kernel = kernel

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
            "kernel": {
                "type": "string",
                "default": "linear",
                "enum": ["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"],
                "description": "Kernel used for PCA"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnKernelPCA":
        """Fit the model from data in X."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = KernelPCA(
            n_components=self.n_components,
            kernel=self.kernel
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnKernelPCA(n_components={self.n_components}, kernel='{self.kernel}')"
