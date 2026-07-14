"""Scikit-Learn manifold learning and non-linear dimensionality reduction."""

import numpy as np
from typing import Dict, List, Any, Optional, Union

from tuiml.base.features import FeatureExtractor, feature_extractor

try:
    from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding
    SKLEARN_AVAILABLE = True
except ImportError:
    TSNE = Isomap = MDS = LocallyLinearEmbedding = None
    SKLEARN_AVAILABLE = False


@feature_extractor(tags=["extraction", "manifold", "tsne", "sklearn"], version="1.0.0")
class SklearnTSNE(FeatureExtractor):
    """Scikit-Learn t-SNE."""

    def __init__(self, n_components: int = 2, perplexity: float = 30.0, metric: str = "euclidean", random_state: Optional[int] = None):
        super().__init__(n_components=n_components)
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.perplexity = perplexity
        self.metric = metric
        self.random_state = random_state
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "n_components": {"type": "integer", "default": 2},
            "perplexity": {"type": "number", "default": 30.0},
            "metric": {"type": "string", "default": "euclidean"}
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X = np.asarray(X, dtype=float)
        self.model_ = TSNE(n_components=self.n_components, perplexity=self.perplexity, metric=self.metric, random_state=self.random_state)
        # t-SNE does not have a transform method, only fit_transform
        self.embedding_ = self.model_.fit_transform(X)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        raise NotImplementedError("t-SNE cannot transform new data. It only embeds the training data. Use fit_transform().")

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.embedding_


@feature_extractor(tags=["extraction", "manifold", "isomap", "sklearn"], version="1.0.0")
class SklearnIsomap(FeatureExtractor):
    """Scikit-Learn Isomap."""

    def __init__(self, n_components: int = 2, n_neighbors: int = 5, metric: str = "minkowski"):
        super().__init__(n_components=n_components)
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "n_components": {"type": "integer", "default": 2},
            "n_neighbors": {"type": "integer", "default": 5},
            "metric": {"type": "string", "default": "minkowski"}
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X = np.asarray(X, dtype=float)
        self.model_ = Isomap(n_components=self.n_components, n_neighbors=self.n_neighbors, metric=self.metric)
        self.model_.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.transform(np.asarray(X, dtype=float))


@feature_extractor(tags=["extraction", "manifold", "mds", "sklearn"], version="1.0.0")
class SklearnMDS(FeatureExtractor):
    """Scikit-Learn Multi-Dimensional Scaling."""

    def __init__(self, n_components: int = 2, metric: bool = True, random_state: Optional[int] = None):
        super().__init__(n_components=n_components)
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.metric_scaling = metric
        self.random_state = random_state
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "n_components": {"type": "integer", "default": 2},
            "metric": {"type": "boolean", "default": True}
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X = np.asarray(X, dtype=float)
        self.model_ = MDS(n_components=self.n_components, metric=self.metric_scaling, random_state=self.random_state)
        self.embedding_ = self.model_.fit_transform(X)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        raise NotImplementedError("MDS cannot transform new data. Use fit_transform().")

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.embedding_


@feature_extractor(tags=["extraction", "manifold", "lle", "sklearn"], version="1.0.0")
class SklearnLocallyLinearEmbedding(FeatureExtractor):
    """Scikit-Learn Locally Linear Embedding."""

    def __init__(self, n_components: int = 2, n_neighbors: int = 5, method: str = "standard", random_state: Optional[int] = None):
        super().__init__(n_components=n_components)
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.n_neighbors = n_neighbors
        self.method = method
        self.random_state = random_state
        self.model_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "n_components": {"type": "integer", "default": 2},
            "n_neighbors": {"type": "integer", "default": 5},
            "method": {"type": "string", "default": "standard", "enum": ["standard", "hessian", "modified", "ltsa"]}
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X = np.asarray(X, dtype=float)
        self.model_ = LocallyLinearEmbedding(n_components=self.n_components, n_neighbors=self.n_neighbors, method=self.method, random_state=self.random_state)
        self.model_.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.transform(np.asarray(X, dtype=float))
