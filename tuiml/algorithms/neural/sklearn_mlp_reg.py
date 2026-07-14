"""Scikit-Learn Multi-Layer Perceptron wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Regressor, regressor

try:
    from sklearn.neural_network import MLPRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    MLPRegressor = None
    SKLEARN_AVAILABLE = False


@regressor(tags=["neural", "mlp", "sklearn"], version="1.0.0")
class SklearnMLPRegressor(Regressor):
    """Scikit-Learn MLP Regressor."""

    def __init__(self, hidden_layer_sizes: tuple = (100,), activation: str = "relu", solver: str = "adam", alpha: float = 0.0001, max_iter: int = 200, random_state: Optional[int] = None):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "hidden_layer_sizes": {"type": "array", "items": {"type": "integer"}, "default": [100]},
            "activation": {"type": "string", "default": "relu", "enum": ["identity", "logistic", "tanh", "relu"]},
            "solver": {"type": "string", "default": "adam", "enum": ["lbfgs", "sgd", "adam"]},
            "alpha": {"type": "number", "default": 0.0001},
            "max_iter": {"type": "integer", "default": 200}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Algorithm":
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        hidden_layers = tuple(self.hidden_layer_sizes) if isinstance(self.hidden_layer_sizes, list) else self.hidden_layer_sizes
        self.model_ = MLPRegressor(hidden_layer_sizes=hidden_layers, activation=self.activation, solver=self.solver, alpha=self.alpha, max_iter=self.max_iter, random_state=self.random_state)
        self.model_.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.predict(np.asarray(X, dtype=float))
