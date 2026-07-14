"""Scikit-Learn LatentDirichletAllocation wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.preprocessing import Transformer, transformer

try:
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    LatentDirichletAllocation = None
    SKLEARN_AVAILABLE = False

@transformer(tags=["decomposition", "sklearn"], version="1.0.0")
class SklearnLatentDirichletAllocation(Transformer):
    """
    Latent Dirichlet Allocation using Scikit-Learn.

    Latent Dirichlet Allocation with online variational Bayes algorithm.

    Parameters
    ----------
    n_components : int, default=10
        Number of topics.
    learning_method : {'batch', 'online'}, default='batch'
        Method used to update _component.
    max_iter : int, default=10
        The maximum number of iterations.
    """

    def __init__(
        self,
        n_components: int = 10,
        learning_method: str = "batch",
        max_iter: int = 10
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_components = n_components
        self.learning_method = learning_method
        self.max_iter = max_iter

        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_components": {
                "type": "integer",
                "default": 10,
                "description": "Number of topics"
            },
            "learning_method": {
                "type": "string",
                "default": "batch",
                "enum": ["batch", "online"],
                "description": "Method used to update _component"
            },
            "max_iter": {
                "type": "integer",
                "default": 10,
                "description": "The maximum number of iterations"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnLatentDirichletAllocation":
        """Learn model for the data X."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = LatentDirichletAllocation(
            n_components=self.n_components,
            learning_method=self.learning_method,
            max_iter=self.max_iter
        )
        self.model_.fit(X)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data X according to the fitted model."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model_.transform(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnLatentDirichletAllocation(n_components={self.n_components})"
