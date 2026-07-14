"""Scikit-Learn Gradient Boosting Classifier wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier

try:
    from sklearn.ensemble import GradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    GradientBoostingClassifier = None
    SKLEARN_AVAILABLE = False

@classifier(tags=["ensemble", "trees", "sklearn"], version="1.0.0")
class SklearnGradientBoostingClassifier(Classifier):
    """
    Gradient Boosting for classification using Scikit-Learn.

    GB builds an additive model in a forward stage-wise fashion; it allows for
    the optimization of arbitrary differentiable loss functions.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to perform.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree.
    max_depth : int, default=3
        Maximum depth of the individual regression estimators.
    random_state : int, optional
        Controls the random seed given to each Tree estimator at each boosting iteration.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: Optional[int] = None
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        self.model_ = None
        self.classes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_estimators": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "The number of boosting stages to perform"
            },
            "learning_rate": {
                "type": "number",
                "default": 0.1,
                "minimum": 0.0,
                "exclusiveMinimum": True,
                "description": "Learning rate shrinks the contribution of each tree"
            },
            "max_depth": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "description": "Maximum depth of the individual regression estimators"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "nominal", "missing_values", "multi_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(M * n * d) time, O(M * trees) space where M=stages, d=features"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. "
            "Annals of statistics, 1189-1232."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnGradientBoostingClassifier":
        """Fit the gradient boosting model."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model_.fit(X, y)

        self.classes_ = self.model_.classes_
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class for X."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.predict_proba(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnGradientBoostingClassifier(n_estimators={self.n_estimators}, learning_rate={self.learning_rate})"
