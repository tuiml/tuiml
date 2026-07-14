"""Scikit-Learn OrdinalEncoder wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.preprocessing import OrdinalEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    OrdinalEncoder = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["encoding", "sklearn"], version="1.0.0")
class SklearnOrdinalEncoder(Transformer):
    """
    OrdinalEncoder wrapper using Scikit-Learn.

    Encode categorical features as an integer array.

    Parameters
    ----------
    handle_unknown : {'error', 'use_encoded_value'}, default='error'
        When set to 'error' an error will be raised in case an unknown categorical feature is present during transform.
    unknown_value : int or np.nan, default=None
        When the parameter handle_unknown is set to 'use_encoded_value', this parameter is required.
    """

    def __init__(
        self,
        handle_unknown: str = "error",
        unknown_value: Optional[float] = None
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "handle_unknown": {
                "type": "string",
                "default": "error",
                "enum": ["error", "use_encoded_value"],
                "description": "How to handle unknown categories"
            },
            "unknown_value": {
                "type": ["number", "null"],
                "default": None,
                "description": "Value to use for unknown categories if handle_unknown is 'use_encoded_value'"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "nominal", "missing_values"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnOrdinalEncoder":
        """Fit the OrdinalEncoder to X."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = OrdinalEncoder(
            handle_unknown=self.handle_unknown,
            unknown_value=self.unknown_value
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to ordinal codes."""
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Convert the data back to the original representation."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.inverse_transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnOrdinalEncoder(handle_unknown='{self.handle_unknown}')"
