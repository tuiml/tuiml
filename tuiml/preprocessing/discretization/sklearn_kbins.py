"""Scikit-Learn KBinsDiscretizer wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.preprocessing import KBinsDiscretizer
    SKLEARN_AVAILABLE = True
except ImportError:
    KBinsDiscretizer = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["discretization", "sklearn"], version="1.0.0")
class SklearnKBinsDiscretizer(Transformer):
    """
    KBinsDiscretizer wrapper using Scikit-Learn.

    Bin continuous data into intervals.

    Parameters
    ----------
    n_bins : int or array-like of shape (n_features,), default=5
        The number of bins to produce. Raises ValueError if n_bins < 2.
    encode : {'onehot', 'onehot-dense', 'ordinal'}, default='onehot'
        Method used to encode the transformed result.
    strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy used to define the widths of the bins.
    """

    def __init__(
        self,
        n_bins: int = 5,
        encode: str = "onehot",
        strategy: str = "quantile"
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_bins": {
                "type": "integer",
                "default": 5,
                "minimum": 2,
                "description": "The number of bins to produce"
            },
            "encode": {
                "type": "string",
                "default": "onehot",
                "enum": ["onehot", "onehot-dense", "ordinal"],
                "description": "Method used to encode the transformed result"
            },
            "strategy": {
                "type": "string",
                "default": "quantile",
                "enum": ["uniform", "quantile", "kmeans"],
                "description": "Strategy used to define the widths of the bins"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnKBinsDiscretizer":
        """Fit the estimator."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode=self.encode,
            strategy=self.strategy
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Discretize the data."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        res = self.model_.transform(X)
        if hasattr(res, "toarray"):
            res = res.toarray()
        return res

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnKBinsDiscretizer(n_bins={self.n_bins}, encode='{self.encode}', strategy='{self.strategy}')"
