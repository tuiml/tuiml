"""Scikit-Learn SVC wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier

try:
    from sklearn.svm import SVC as Sklearn_SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    Sklearn_SVC = None
    SKLEARN_AVAILABLE = False

@classifier(tags=["svm", "sklearn"], version="1.0.0")
class SklearnSVC(Classifier):
    """
    C-Support Vector Classification using Scikit-Learn.

    The implementation is based on libsvm. The fit time scales at least
    quadratically with the number of samples and may be impractical
    beyond tens of thousands of samples.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
    kernel : {"linear", "poly", "rbf", "sigmoid"}, default="rbf"
        Specifies the kernel type to be used in the algorithm.
    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
    gamma : {"scale", "auto"} or float, default="scale"
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    random_state : int, optional
        Controls the pseudo random number generation for shuffling the data for
        probability estimates.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: Any = "scale",
        random_state: Optional[int] = None
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.random_state = random_state

        self.model_ = None
        self.classes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "C": {
                "type": "number",
                "default": 1.0,
                "minimum": 0,
                "exclusiveMinimum": True,
                "description": "Regularization parameter"
            },
            "kernel": {
                "type": "string",
                "default": "rbf",
                "enum": ["linear", "poly", "rbf", "sigmoid"],
                "description": "Kernel type"
            },
            "degree": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "description": "Degree of the polynomial kernel"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric", "multi_class"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n^2 * d) time, O(n^2) space where n=samples, d=features"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Chang, C. C., & Lin, C. J. (2011). LIBSVM: a library for support "
            "vector machines."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnSVC":
        """Fit the SVM model according to the given training data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = Sklearn_SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            probability=True,  # To allow predict_proba
            random_state=self.random_state
        )
        self.model_.fit(X, y)

        self.classes_ = self.model_.classes_
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Perform classification on samples in X."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute probabilities of possible outcomes for samples in X."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.predict_proba(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnSVC(C={self.C}, kernel='{self.kernel}')"
