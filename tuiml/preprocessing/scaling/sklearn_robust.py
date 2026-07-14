"""Scikit-Learn RobustScaler wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.preprocessing import RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    RobustScaler = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["scaling", "sklearn"], version="1.0.0")
class SklearnRobustScaler(Transformer):
    """
    RobustScaler wrapper using Scikit-Learn.

    Scale features using statistics that are robust to outliers.

    Parameters
    ----------
    with_centering : bool, default=True
        If True, center the data before scaling.
    with_scaling : bool, default=True
        If True, scale the data to interquartile range.
    """

    def __init__(
        self,
        with_centering: bool = True,
        with_scaling: bool = True
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.with_centering = with_centering
        self.with_scaling = with_scaling

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "with_centering": {
                "type": "boolean",
                "default": True,
                "description": "Center the data before scaling"
            },
            "with_scaling": {
                "type": "boolean",
                "default": True,
                "description": "Scale the data to interquartile range"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "missing_values"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnRobustScaler":
        """Compute the median and quantiles to be used for scaling."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = RobustScaler(
            with_centering=self.with_centering,
            with_scaling=self.with_scaling
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Center and scale the data."""
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
        return f"SklearnRobustScaler(with_centering={self.with_centering}, with_scaling={self.with_scaling})"
