"""Scikit-Learn OneHotEncoder wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.preprocessing import OneHotEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    OneHotEncoder = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["encoding", "sklearn"], version="1.0.0")
class SklearnOneHotEncoder(Transformer):
    """
    OneHotEncoder wrapper using Scikit-Learn.

    Encode categorical features as a one-hot numeric array.

    Parameters
    ----------
    drop : {'first', 'if_binary'} or an array-like of shape (n_features,), default=None
        Specifies a methodology to use to drop one of the categories per feature.
    sparse_output : bool, default=False
        Will return sparse matrix if set True else will return an array.
    handle_unknown : {'error', 'ignore', 'infrequent_if_exist'}, default='error'
        Whether to raise an error or ignore if an unknown categorical feature is present during transform.
    """

    def __init__(
        self,
        drop: Optional[str] = None,
        sparse_output: bool = False,
        handle_unknown: str = "error"
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.drop = drop
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "drop": {
                "type": ["string", "null"],
                "default": None,
                "enum": ["first", "if_binary", None],
                "description": "Specifies a methodology to use to drop one of the categories"
            },
            "sparse_output": {
                "type": "boolean",
                "default": False,
                "description": "Will return sparse matrix if set True else will return an array"
            },
            "handle_unknown": {
                "type": "string",
                "default": "error",
                "enum": ["error", "ignore", "infrequent_if_exist"],
                "description": "Whether to raise an error or ignore if an unknown categorical feature is present"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "nominal", "missing_values"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnOneHotEncoder":
        """Fit OneHotEncoder to X."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = OneHotEncoder(
            drop=self.drop,
            sparse_output=self.sparse_output,
            handle_unknown=self.handle_unknown
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X using one-hot encoding."""
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnOneHotEncoder(drop={self.drop}, sparse_output={self.sparse_output}, handle_unknown='{self.handle_unknown}')"
