"""Scikit-Learn StandardScaler wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    StandardScaler = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["scaling", "sklearn"], version="1.0.0")
class SklearnStandardScaler(Transformer):
    """
    StandardScaler wrapper using Scikit-Learn.

    Standardize features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance.
    """

    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.with_mean = with_mean
        self.with_std = with_std

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "with_mean": {
                "type": "boolean",
                "default": True,
                "description": "Center the data before scaling"
            },
            "with_std": {
                "type": "boolean",
                "default": True,
                "description": "Scale the data to unit variance"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "missing_values"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnStandardScaler":
        """Compute the mean and std to be used for later scaling."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = StandardScaler(
            with_mean=self.with_mean,
            with_std=self.with_std
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Perform standardization by centering and scaling."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Scale back the data to the original representation."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.inverse_transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnStandardScaler(with_mean={self.with_mean}, with_std={self.with_std})"
