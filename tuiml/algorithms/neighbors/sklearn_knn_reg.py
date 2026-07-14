"""Scikit-Learn K-Neighbors Regressor wrapper."""

import numpy as np
from typing import Dict, List, Any

from tuiml.base.algorithms import Regressor, regressor

try:
    from sklearn.neighbors import KNeighborsRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    KNeighborsRegressor = None
    SKLEARN_AVAILABLE = False

@regressor(tags=["neighbors", "sklearn"], version="1.0.0")
class SklearnKNeighborsRegressor(Regressor):
    """
    Regression based on k-nearest neighbors using Scikit-Learn.

    The target is predicted by local interpolation of the targets
    associated of the nearest neighbors in the training set.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for queries.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.
    metric : str, default='minkowski'
        Metric to use for distance computation.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "minkowski"
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_neighbors": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "description": "Number of neighbors to use"
            },
            "weights": {
                "type": "string",
                "default": "uniform",
                "enum": ["uniform", "distance"],
                "description": "Weight function used in prediction"
            },
            "metric": {
                "type": "string",
                "default": "minkowski",
                "description": "Metric to use for distance computation"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "regression"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(1) training time, O(n * d) prediction time where d=features"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Altman, N. S. (1992). An introduction to kernel and nearest-neighbor nonparametric regression. "
            "The American Statistician, 46(3), 175-185."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnKNeighborsRegressor":
        """Fit the k-nearest neighbors regressor from the training dataset."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric
        )
        self.model_.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target for the provided data."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.predict(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnKNeighborsRegressor(n_neighbors={self.n_neighbors}, weights='{self.weights}')"
