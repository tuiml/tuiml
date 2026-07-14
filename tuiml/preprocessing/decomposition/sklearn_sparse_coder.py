"""Scikit-Learn SparseCoder wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.decomposition import SparseCoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SparseCoder = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["decomposition", "sklearn"], version="1.0.0")
class SklearnSparseCoder(Transformer):
    """
    Sparse coding using Scikit-Learn.

    Finds a sparse representation of data against a fixed, precomputed
    dictionary.

    Parameters
    ----------
    dictionary : ndarray of shape (n_components, n_features)
        The dictionary atoms used for sparse coding.
    transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}, default='omp'
        Algorithm used to transform the data.
    """

    def __init__(
        self,
        dictionary: Optional[np.ndarray] = None,
        transform_algorithm: str = "omp"
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.dictionary = dictionary
        self.transform_algorithm = transform_algorithm

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "dictionary": {
                "type": ["array", "null"],
                "default": None,
                "description": "The dictionary atoms used for sparse coding"
            },
            "transform_algorithm": {
                "type": "string",
                "default": "omp",
                "enum": ["lasso_lars", "lasso_cd", "lars", "omp", "threshold"],
                "description": "Algorithm used to transform the data"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnSparseCoder":
        """Do nothing and return the estimator unchanged."""
        if self.dictionary is None:
            raise ValueError("SparseCoder requires a predefined dictionary.")
            
        self.model_ = SparseCoder(
            dictionary=np.asarray(self.dictionary),
            transform_algorithm=self.transform_algorithm
        )
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
        return f"SklearnSparseCoder(transform_algorithm='{self.transform_algorithm}')"
