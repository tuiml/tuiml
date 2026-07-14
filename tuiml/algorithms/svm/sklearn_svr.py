"""Scikit-Learn Support Vector Regressor wrapper."""

import numpy as np
from typing import Dict, List, Any

from tuiml.base.algorithms import Regressor, regressor

try:
    from sklearn.svm import SVR
    SKLEARN_AVAILABLE = True
except ImportError:
    SVR = None
    SKLEARN_AVAILABLE = False

@regressor(tags=["svm", "sklearn"], version="1.0.0")
class SklearnSVR(Regressor):
    """
    Epsilon-Support Vector Regression using Scikit-Learn.

    The free parameters in the model are C and epsilon.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable, default='rbf'
        Specifies the kernel type to be used in the algorithm.
    C : float, default=1.0
        Regularization parameter.
    epsilon : float, default=0.1
        Epsilon in the epsilon-SVR model.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "kernel": {
                "type": "string",
                "default": "rbf",
                "enum": ["linear", "poly", "rbf", "sigmoid"],
                "description": "Specifies the kernel type to be used"
            },
            "C": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.0,
                "exclusiveMinimum": True,
                "description": "Regularization parameter"
            },
            "epsilon": {
                "type": "number",
                "default": 0.1,
                "minimum": 0.0,
                "description": "Epsilon in the epsilon-SVR model"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "regression"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n^2 * d) to O(n^3 * d) training time, O(n_support * d) prediction time"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Smola, A. J., & Schölkopf, B. (2004). "
            "A tutorial on support vector regression. Statistics and computing, 14(3), 199-222."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnSVR":
        """Fit the SVM model according to the given training data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon
        )
        self.model_.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform regression on samples in X."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.predict(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnSVR(kernel='{self.kernel}', C={self.C})"
