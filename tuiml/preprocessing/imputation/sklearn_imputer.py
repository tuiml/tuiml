"""Scikit-Learn SimpleImputer wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SimpleImputer = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["imputation", "sklearn"], version="1.0.0")
class SklearnSimpleImputer(Transformer):
    """
    SimpleImputer wrapper using Scikit-Learn.

    Imputation transformer for completing missing values.

    Parameters
    ----------
    strategy : str, default='mean'
        The imputation strategy. If 'mean', then replace missing values using the mean along each column.
        Can be 'mean', 'median', 'most_frequent', or 'constant'.
    fill_value : str or numerical value, default=None
        When strategy == 'constant', fill_value is used to replace all occurrences of missing_values.
    add_indicator : bool, default=False
        If True, a MissingIndicator transform will stack onto output of the imputer's transform.
    """

    def __init__(
        self,
        strategy: str = "mean",
        fill_value: Optional[Any] = None,
        add_indicator: bool = False
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "strategy": {
                "type": "string",
                "default": "mean",
                "enum": ["mean", "median", "most_frequent", "constant"],
                "description": "The imputation strategy"
            },
            "add_indicator": {
                "type": "boolean",
                "default": False,
                "description": "Add a MissingIndicator transform"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "nominal", "missing_values"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnSimpleImputer":
        """Fit the imputer on X."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = SimpleImputer(
            strategy=self.strategy,
            fill_value=self.fill_value,
            add_indicator=self.add_indicator
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute all missing values in X."""
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnSimpleImputer(strategy='{self.strategy}')"
