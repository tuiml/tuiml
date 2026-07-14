"""Scikit-Learn Decision Tree Regressor wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Regressor, regressor

try:
    from sklearn.tree import DecisionTreeRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    DecisionTreeRegressor = None
    SKLEARN_AVAILABLE = False

@regressor(tags=["trees", "sklearn"], version="1.0.0")
class SklearnDecisionTreeRegressor(Regressor):
    """
    A decision tree regressor using Scikit-Learn.

    Parameters
    ----------
    max_depth : int, default=None
        The maximum depth of the tree.
    random_state : int, optional
        Controls the randomness of the estimator.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.max_depth = max_depth
        self.random_state = random_state

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
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
        return "O(n * log(n) * d) time, O(nodes) space where d=features"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). "
            "Classification and regression trees. Monterey, CA: Wadsworth & Brooks/Cole Advanced Books & Software."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnDecisionTreeRegressor":
        """Build a decision tree regressor from the training set (X, y)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = DecisionTreeRegressor(
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model_.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class or regression value for X."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.predict(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnDecisionTreeRegressor(max_depth={self.max_depth})"
