"""Scikit-Learn PolynomialFeatures wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.preprocessing import PolynomialFeatures
    SKLEARN_AVAILABLE = True
except ImportError:
    PolynomialFeatures = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["preprocessing", "sklearn"], version="1.0.0")
class SklearnPolynomialFeatures(Transformer):
    """
    PolynomialFeatures wrapper using Scikit-Learn.

    Generate polynomial and interaction features.

    Parameters
    ----------
    degree : int, default=2
        The degree of the polynomial features.
    interaction_only : bool, default=False
        If True, only interaction features are produced.
    include_bias : bool, default=True
        If True, then include a bias column.
    """

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = True
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "degree": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "The degree of the polynomial features"
            },
            "interaction_only": {
                "type": "boolean",
                "default": False,
                "description": "If True, only interaction features are produced"
            },
            "include_bias": {
                "type": "boolean",
                "default": True,
                "description": "If True, then include a bias column"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnPolynomialFeatures":
        """Compute number of output features."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to polynomial features."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnPolynomialFeatures(degree={self.degree}, interaction_only={self.interaction_only}, include_bias={self.include_bias})"
