"""Scikit-Learn Normalizer wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.preprocessing import Normalizer
    SKLEARN_AVAILABLE = True
except ImportError:
    Normalizer = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["scaling", "sklearn"], version="1.0.0")
class SklearnNormalizer(Transformer):
    """
    Normalizer wrapper using Scikit-Learn.

    Normalize samples individually to unit norm.

    Parameters
    ----------
    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non zero sample.
    """

    def __init__(
        self,
        norm: str = "l2"
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.norm = norm
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "norm": {
                "type": "string",
                "default": "l2",
                "enum": ["l1", "l2", "max"],
                "description": "The norm to use to normalize each non zero sample"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnNormalizer":
        """Do nothing and return the estimator unchanged."""
        self.model_ = Normalizer(norm=self.norm)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale each non zero row of X to unit norm."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnNormalizer(norm='{self.norm}')"
