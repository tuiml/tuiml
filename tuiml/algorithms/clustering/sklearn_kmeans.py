"""Scikit-Learn KMeans clustering wrapper."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Clusterer, clusterer

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    KMeans = None
    SKLEARN_AVAILABLE = False

@clusterer(tags=["partitional", "centroid-based", "sklearn"], version="1.0.0")
class SklearnKMeansClusterer(Clusterer):
    """
    K-Means clustering algorithm using Scikit-Learn.

    Partitions data into k clusters by iteratively assigning points to
    the nearest centroid and updating centroids to the mean of assigned points.
    This implementation leverages Scikit-Learn's optimized KMeans.

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form.
    init : {"k-means++", "random"}, default="k-means++"
        Method for initialization.
    n_init : int or str, default="auto"
        Number of times the k-means algorithm will be run.
    max_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-4
        Relative tolerance.
    random_state : int, optional
        Determines random number generation.

    Attributes
    ----------
    cluster_centers_ : np.ndarray
        Coordinates of cluster centers.
    labels_ : np.ndarray
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        init: str = "k-means++",
        n_init: Any = "auto",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None
    ):
        super().__init__()
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
            
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.model_ = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "n_clusters": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description": "Number of clusters"
            },
            "init": {
                "type": "string",
                "default": "k-means++",
                "enum": ["k-means++", "random"],
                "description": "Initialization method"
            },
            "max_iter": {
                "type": "integer",
                "default": 300,
                "minimum": 1,
                "description": "Maximum iterations"
            },
            "tol": {
                "type": "number",
                "default": 1e-4,
                "minimum": 0,
                "description": "Convergence tolerance"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return algorithm capabilities."""
        return ["numeric"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(n * k * d * i) time, O(n * d + k * d) space"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Arthur, D. & Vassilvitskii, S. (2007). k-means++: the advantages "
            "of careful seeding. SODA '07, 1027-1035."
        ]

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnKMeansClusterer":
        """Compute k-means clustering."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model_ = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state
        )
        self.model_.fit(X)

        self.cluster_centers_ = self.model_.cluster_centers_
        self.labels_ = self.model_.labels_
        self.inertia_ = self.model_.inertia_
        self.n_iter_ = self.model_.n_iter_
        self.n_clusters_ = self.n_clusters

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.predict(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to a cluster-distance space."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model_.transform(X)

    def __repr__(self) -> str:
        """String representation."""
        if self._is_fitted:
            return (f"SklearnKMeansClusterer(n_clusters={self.n_clusters_}, "
                   f"inertia={self.inertia_:.2f}, n_iter={self.n_iter_})")
        return f"SklearnKMeansClusterer(n_clusters={self.n_clusters})"
