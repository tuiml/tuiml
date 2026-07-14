"""Scikit-Learn Binarizer wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.preprocessing import Binarizer
    SKLEARN_AVAILABLE = True
except ImportError:
    Binarizer = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["preprocessing", "sklearn"], version="1.0.0")
class SklearnBinarizer(Transformer):
    """
    Binarizer wrapper using Scikit-Learn.

    Binarize data (set feature values to 0 or 1) according to a threshold.

    Parameters
    ----------
    threshold : float, default=0.0
        Feature values below or equal to this are replaced by 0, above it by 1.
    """

    def __init__(
        self,
        threshold: float = 0.0
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.threshold = threshold
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "threshold": {
                "type": "number",
                "default": 0.0,
                "description": "Feature values below or equal to this are replaced by 0, above it by 1"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnBinarizer":
        """Do nothing and return the estimator unchanged."""
        self.model_ = Binarizer(threshold=self.threshold)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Binarize each element of X."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnBinarizer(threshold={self.threshold})"
