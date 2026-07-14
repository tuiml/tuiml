"""Scikit-Learn Multi-layer Perceptron Classifier wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier

try:
    from sklearn.neural_network import MLPClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    MLPClassifier = None
    SKLEARN_AVAILABLE = False

@classifier(tags=["neural", "sklearn"], version="1.0.0")
class SklearnMLPClassifier(Classifier):
    """
    Multi-layer Perceptron classifier using Scikit-Learn.

    This model optimizes the log-loss function using LBFGS or stochastic gradient descent.

    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.
    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer.
    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization.
    max_iter : int, default=200
        Maximum number of iterations.
    random_state : int, optional
        Determines random number generation for weights and bias initialization.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        activation: str = "relu",
        solver: str = "adam",
        max_iter: int = 200,
        random_state: Optional[int] = None
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

        self.model_ = None
        self.classes_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "hidden_layer_sizes": {
                "type": "array",
                "items": {"type": "integer"},
                "default": [100],
                "description": "Sizes of hidden layers"
            },
            "activation": {
                "type": "string",
                "default": "relu",
                "enum": ["identity", "logistic", "tanh", "relu"],
                "description": "Activation function for the hidden layer"
            },
            "solver": {
                "type": "string",
                "default": "adam",
                "enum": ["lbfgs", "sgd", "adam"],
                "description": "The solver for weight optimization"
            },
            "max_iter": {
                "type": "integer",
                "default": 200,
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
        return "O(n * m * d * i) time where m=hidden_nodes, i=iterations. O(m * d) space"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Hinton, G. E. (1989). Connectionist learning procedures. "
            "Artificial intelligence, 40(1-3), 185-234."
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnMLPClassifier":
        """Fit the model to data matrix X and target(s) y."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_layer_sizes),
            activation=self.activation,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.model_.fit(X, y)

        self.classes_ = self.model_.classes_
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the multi-layer perceptron classifier."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability estimates."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.predict_proba(X)

    def __repr__(self) -> str:
        """String representation."""
        return f"SklearnMLPClassifier(hidden_layer_sizes={self.hidden_layer_sizes}, activation='{self.activation}')"
