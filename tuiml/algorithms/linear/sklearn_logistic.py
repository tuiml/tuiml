"""Scikit-Learn Logistic Regression wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier

try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    LogisticRegression = None
    SKLEARN_AVAILABLE = False

@classifier(tags=["linear", "sklearn"], version="1.0.0")
class SklearnLogisticRegression(Classifier):
    """
    Logistic Regression classifier using Scikit-Learn.

    In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    scheme if the 'multi_class' option is set to 'ovr', and uses the
    cross-entropy loss if the 'multi_class' option is set to 'multinomial'.

    Parameters
    ----------
    penalty : {"l1", "l2", "elasticnet", None}, default="l2"
        Specify the norm of the penalty.
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.
    random_state : int, optional
        Used when solver == 'sag', 'saga' or 'liblinear' to shuffle the data.
    """

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        max_iter: int = 100,
        random_state: Optional[int] = None
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

        self.model_ = None
        self.classes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "penalty": {
                "type": ["string", "null"],
                "default": "l2",
                "enum": ["l1", "l2", "elasticnet", None],
                "description": "Specify the norm of the penalty"
            },
            "C": {
                "type": "number",
                "default": 1.0,
                "minimum": 0,
                "exclusiveMinimum": True,
                "description": "Inverse of regularization strength"
            },
            "max_iter": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Maximum number of iterations"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "multi_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * d) time, O(d) space where d=features"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Fan, R. E., Chang, K. W., Hsieh, C. J., Wang, X. R., & Lin, C. J. "
            "(2008). LIBLINEAR: A library for large linear classification."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnLogisticRegression":
        """Fit the model according to the given training data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver='liblinear' if self.penalty == 'l1' else 'lbfgs'
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
        return f"SklearnLogisticRegression(penalty='{self.penalty}', C={self.C})"
