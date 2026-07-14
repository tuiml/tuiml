"""Scikit-Learn Linear Regression wrapper."""

import numpy as np
from typing import Dict, List, Any

from tuiml.base.algorithms import Regressor, regressor

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    LinearRegression = None
    SKLEARN_AVAILABLE = False

@regressor(tags=["linear", "sklearn"], version="1.0.0")
class SklearnLinearRegression(Regressor):
    """
    Ordinary least squares Linear Regression using Scikit-Learn.

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """

    def __init__(
        self,
        fit_intercept: bool = True
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.fit_intercept = fit_intercept

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "fit_intercept": {
                "type": "boolean",
                "default": True,
                "description": "Whether to calculate the intercept for this model"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "regression"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * d^2) time, O(d) space where d=features"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnLinearRegression":
        """Fit linear model."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = LinearRegression(
            fit_intercept=self.fit_intercept
        )
        self.model_.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.predict(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnLinearRegression(fit_intercept={self.fit_intercept})"
