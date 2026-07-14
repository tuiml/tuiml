"""Scikit-Learn MiniBatchDictionaryLearning wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.decomposition import MiniBatchDictionaryLearning
    SKLEARN_AVAILABLE = True
except ImportError:
    MiniBatchDictionaryLearning = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["decomposition", "sklearn"], version="1.0.0")
class SklearnMiniBatchDictionaryLearning(Transformer):
    """
    Mini-batch dictionary learning using Scikit-Learn.

    Finds a dictionary (a set of atoms) that can best be used to represent data
    using a sparse code.

    Parameters
    ----------
    n_components : int, default=None
        Number of dictionary elements to extract.
    alpha : float, default=1.0
        Sparsity controlling parameter.
    batch_size : int, default=3
        Number of samples in each mini-batch.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        alpha: float = 1.0,
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
                "description": "Number of dictionary elements to extract"
            },
            "alpha": {
                "type": "number",
                "default": 1.0,
                "description": "Sparsity controlling parameter"
            },
            "batch_size": {
                "type": "integer",
                "default": 3,
                "description": "Number of samples in each mini-batch"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnMiniBatchDictionaryLearning":
        """Fit the model from data in X."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = MiniBatchDictionaryLearning(
            n_components=self.n_components,
            alpha=self.alpha,
            batch_size=self.batch_size
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode the data as a sparse combination of the dictionary atoms."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnMiniBatchDictionaryLearning(n_components={self.n_components}, alpha={self.alpha})"
