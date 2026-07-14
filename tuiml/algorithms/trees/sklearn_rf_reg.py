"""Scikit-Learn Random Forest Regressor wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Regressor, regressor

try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestRegressor = None
    SKLEARN_AVAILABLE = False

@regressor(tags=["ensemble", "trees", "sklearn"], version="1.0.0")
class SklearnRandomForestRegressor(Regressor):
    """
    A random forest regressor using Scikit-Learn.

    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : int, default=None
        The maximum depth of the tree.
    random_state : int, optional
        Controls both the randomness of the bootstrapping of the samples used
        when building trees and the sampling of the features.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_estimators": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "The number of trees in the forest"
            },
            "max_depth": {
                "type": ["integer", "null"],
                "default": None,
                "description": "The maximum depth of the tree"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "regression"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(M * n * log(n) * d) time, O(M * trees) space where M=trees, d=features"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnRandomForestRegressor":
        """Build a forest of trees from the training set (X, y)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model_.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression target for X."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.predict(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnRandomForestRegressor(n_estimators={self.n_estimators}, max_depth={self.max_depth})"
