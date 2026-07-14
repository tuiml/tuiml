"""Scikit-Learn Random Projection wrappers."""

import numpy as np
from typing import Dict, List, Any, Optional, Union

from tuiml.base.features import FeatureExtractor, feature_extractor

try:
    from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
    SKLEARN_AVAILABLE = True
except ImportError:
    GaussianRandomProjection = SparseRandomProjection = None
    SKLEARN_AVAILABLE = False


@feature_extractor(tags=["extraction", "dimensionality_reduction", "random_projection", "sklearn"], version="1.0.0")
class SklearnGaussianRandomProjection(FeatureExtractor):
    """Gaussian Random Projection using Scikit-Learn."""

    def __init__(
        self,
        n_components: Union[str, int] = 'auto',
        eps: float = 0.1,
        random_state: Optional[int] = None
    ):
        super().__init__(n_components=n_components if isinstance(n_components, int) else None)
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.n_components_param = n_components
        self.eps = eps
        self.random_state = random_state
        self.model_ = None
        self.n_components_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "n_components": {"type": ["integer", "string"], "default": "auto"},
            "eps": {"type": "number", "default": 0.1}
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X = np.asarray(X)
        self.model_ = GaussianRandomProjection(
            n_components=self.n_components_param,
            eps=self.eps,
            random_state=self.random_state
        )
        self.model_.fit(X)
        self.n_components_ = self.model_.n_components_
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.transform(np.asarray(X))

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


@feature_extractor(tags=["extraction", "dimensionality_reduction", "random_projection", "sklearn", "sparse"], version="1.0.0")
class SklearnSparseRandomProjection(FeatureExtractor):
    """Sparse Random Projection using Scikit-Learn."""

    def __init__(
        self,
        n_components: Union[str, int] = 'auto',
        density: Union[str, float] = 'auto',
        eps: float = 0.1,
        dense_output: bool = False,
        random_state: Optional[int] = None
    ):
        super().__init__(n_components=n_components if isinstance(n_components, int) else None)
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.n_components_param = n_components
        self.density = density
        self.eps = eps
        self.dense_output = dense_output
        self.random_state = random_state
        self.model_ = None
        self.n_components_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "n_components": {"type": ["integer", "string"], "default": "auto"},
            "density": {"type": ["number", "string"], "default": "auto"},
            "eps": {"type": "number", "default": 0.1},
            "dense_output": {"type": "boolean", "default": False}
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X = np.asarray(X)
        self.model_ = SparseRandomProjection(
            n_components=self.n_components_param,
            density=self.density,
            eps=self.eps,
            dense_output=self.dense_output,
            random_state=self.random_state
        )
        self.model_.fit(X)
        self.n_components_ = self.model_.n_components_
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        out = self.model_.transform(np.asarray(X))
        if hasattr(out, "toarray") and self.dense_output:
            return out.toarray()
        return out

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)
