"""Scikit-Learn clustering algorithms."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Clusterer, clusterer

try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering, OPTICS
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    DBSCAN = AgglomerativeClustering = SpectralClustering = OPTICS = GaussianMixture = None
    SKLEARN_AVAILABLE = False


@clusterer(tags=["density-based", "sklearn"], version="1.0.0")
class SklearnDBSCAN(Clusterer):
    """Scikit-Learn DBSCAN clustering."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean"):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.model_ = None
        self.labels_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "eps": {"type": "number", "default": 0.5},
            "min_samples": {"type": "integer", "default": 5},
            "metric": {"type": "string", "default": "euclidean"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X = np.asarray(X, dtype=float)
        self.model_ = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        raise NotImplementedError("DBSCAN cannot predict on new unseen data. Use .fit() and access .labels_")


@clusterer(tags=["hierarchical", "sklearn"], version="1.0.0")
class SklearnAgglomerativeClustering(Clusterer):
    """Scikit-Learn Agglomerative Clustering."""

    def __init__(self, n_clusters: int = 2, metric: str = "euclidean", linkage: str = "ward"):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage
        self.model_ = None
        self.labels_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "n_clusters": {"type": "integer", "default": 2},
            "metric": {"type": "string", "default": "euclidean"},
            "linkage": {"type": "string", "default": "ward"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X = np.asarray(X, dtype=float)
        self.model_ = AgglomerativeClustering(n_clusters=self.n_clusters, metric=self.metric, linkage=self.linkage)
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        raise NotImplementedError("AgglomerativeClustering cannot predict on new unseen data. Use .fit() and access .labels_")


@clusterer(tags=["spectral", "sklearn"], version="1.0.0")
class SklearnSpectralClustering(Clusterer):
    """Scikit-Learn Spectral Clustering."""

    def __init__(self, n_clusters: int = 8, assign_labels: str = "kmeans", random_state: Optional[int] = None):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.n_clusters = n_clusters
        self.assign_labels = assign_labels
        self.random_state = random_state
        self.model_ = None
        self.labels_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "n_clusters": {"type": "integer", "default": 8},
            "assign_labels": {"type": "string", "default": "kmeans"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X = np.asarray(X, dtype=float)
        self.model_ = SpectralClustering(n_clusters=self.n_clusters, assign_labels=self.assign_labels, random_state=self.random_state)
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        raise NotImplementedError("SpectralClustering cannot predict on new unseen data. Use .fit() and access .labels_")


@clusterer(tags=["density-based", "sklearn"], version="1.0.0")
class SklearnOPTICS(Clusterer):
    """Scikit-Learn OPTICS clustering."""

    def __init__(self, min_samples: int = 5, metric: str = "euclidean"):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.min_samples = min_samples
        self.metric = metric
        self.model_ = None
        self.labels_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "min_samples": {"type": "integer", "default": 5},
            "metric": {"type": "string", "default": "euclidean"}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X = np.asarray(X, dtype=float)
        self.model_ = OPTICS(min_samples=self.min_samples, metric=self.metric)
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        raise NotImplementedError("OPTICS cannot predict on new unseen data. Use .fit() and access .labels_")


@clusterer(tags=["probabilistic", "sklearn"], version="1.0.0")
class SklearnGaussianMixture(Clusterer):
    """Scikit-Learn Gaussian Mixture Model clustering."""

    def __init__(self, n_components: int = 1, covariance_type: str = "full", random_state: Optional[int] = None):
        super().__init__()
        if not SKLEARN_AVAILABLE: raise ImportError("scikit-learn is not installed.")
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model_ = None
        self.labels_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "n_components": {"type": "integer", "default": 1},
            "covariance_type": {"type": "string", "default": "full", "enum": ["full", "tied", "diag", "spherical"]}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric"]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X = np.asarray(X, dtype=float)
        self.model_ = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type, random_state=self.random_state)
        self.model_.fit(X)
        self.labels_ = self.model_.predict(X)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.predict(np.asarray(X, dtype=float))
