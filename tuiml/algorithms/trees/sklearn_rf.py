"""Scikit-Learn Random Forest wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestClassifier = None
    SKLEARN_AVAILABLE = False

@classifier(tags=["ensemble", "trees", "sklearn"], version="1.0.0")
class SklearnRandomForestClassifier(Classifier):
    """
    Random Forest classification algorithm using Scikit-Learn.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split.
    max_depth : int, optional
        The maximum depth of the tree.
    random_state : int, optional
        Controls both the randomness of the bootstrapping of the samples used
        when building trees and the sampling of the features to consider when
        looking for the best split at each node.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_estimators = n_estimators
        self.criterion = criterion
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
                "description": "The number of trees in the forest"
            },
            "criterion": {
                "type": "string",
                "default": "gini",
                "enum": ["gini", "entropy", "log_loss"],
                "description": "The function to measure the quality of a split"
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
        return ["numeric", "nominal", "missing_values", "multi_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(M * d * n log n) time, O(M * trees) space where M=trees, d=features"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnRandomForestClassifier":
        """Build a forest of trees from the training set (X, y)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
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
        return f"SklearnRandomForestClassifier(n_estimators={self.n_estimators}, criterion='{self.criterion}')"
