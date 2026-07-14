"""Scikit-Learn Decision Tree wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier

try:
    from sklearn.tree import DecisionTreeClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    DecisionTreeClassifier = None
    SKLEARN_AVAILABLE = False

@classifier(tags=["trees", "sklearn"], version="1.0.0")
class SklearnDecisionTreeClassifier(Classifier):
    """
    Decision Tree classification algorithm using Scikit-Learn.

    A decision tree classifier creates a model that predicts the value of a
    target variable by learning simple decision rules inferred from the data
    features.

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split.
    max_depth : int, optional
        The maximum depth of the tree.
    random_state : int, optional
        Controls the randomness of the estimator.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state

        self.model_ = None
        self.classes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
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
        return "O(n * d log n) time, O(n * d) space where d=features"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). "
            "Classification and Regression Trees."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnDecisionTreeClassifier":
        """Build a decision tree classifier from the training set (X, y)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = DecisionTreeClassifier(
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
        return f"SklearnDecisionTreeClassifier(criterion='{self.criterion}')"
