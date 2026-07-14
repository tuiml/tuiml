"""Scikit-Learn K-Neighbors Classifier wrapper."""

import numpy as np
from typing import Dict, List, Any

from tuiml.base.algorithms import Classifier, classifier

try:
    from sklearn.neighbors import KNeighborsClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    KNeighborsClassifier = None
    SKLEARN_AVAILABLE = False

@classifier(tags=["neighbors", "sklearn"], version="1.0.0")
class SklearnKNeighborsClassifier(Classifier):
    """
    Classifier implementing the k-nearest neighbors vote using Scikit-Learn.

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
        self.classes_ = None

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
        return ["numeric", "multi_class"]

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnKNeighborsClassifier":
        """Fit the k-nearest neighbors classifier from the training dataset."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric
        )
        self.model_.fit(X, y)

        self.classes_ = self.model_.classes_
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class labels for the provided data."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for the test data X."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.predict_proba(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnKNeighborsClassifier(n_neighbors={self.n_neighbors}, weights='{self.weights}')"
