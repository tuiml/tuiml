"""
Gaussian Blobs data generator.

Generates isotropic Gaussian blobs for clustering.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Union

from tuiml.base.generators import ClusteringGenerator, GeneratedData

class Blobs(ClusteringGenerator):
    """
    Gaussian Blobs data generator.

    Generates data from isotropic Gaussian distributions (blobs).
    This is a common test case for clustering algorithms.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate (total or per cluster).
    n_features : int, default=2
        Number of features (dimensions).
    n_clusters : int, default=3
        Number of clusters (blobs).
    cluster_std : float or list of float, default=1.0
        Standard deviation of clusters.
    centers : np.ndarray or None, default=None
        Array of cluster centers (optional).
    center_box : tuple, default=(-10.0, 10.0)
        Bounding box for randomly generated centers.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Examples
    --------
    >>> gen = Blobs(n_samples=1000, n_clusters=4)
    >>> data = gen.generate()
    >>> X, y = gen(return_X_y=True)
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 2,
        n_clusters: int = 3,
        cluster_std: Union[float, List[float]] = 1.0,
        centers: Optional[np.ndarray] = None,
        center_box: tuple = (-10.0, 10.0),
        shuffle: bool = True,
        random_state: Optional[int] = None
    ):
        """Initialize Blobs generator.

        Parameters
        ----------
        n_samples : int, default=100
            Total number of samples to generate.
        n_features : int, default=2
            Number of features (dimensions).
        n_clusters : int, default=3
            Number of Gaussian blobs.
        cluster_std : float or list of float, default=1.0
            Standard deviation of each cluster.
        centers : np.ndarray or None, default=None
            Array of cluster center coordinates. If None, centers are
            randomly generated within ``center_box``.
        center_box : tuple, default=(-10.0, 10.0)
            Bounding box for randomly generated centers.
        shuffle : bool, default=True
            Whether to shuffle the samples.
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        super().__init__(n_samples, n_features, n_clusters, random_state)
        self.cluster_std = cluster_std
        self.centers = centers
        self.center_box = center_box
        self.shuffle = shuffle

    def generate(self) -> GeneratedData:
        """Generate Gaussian blobs.

        Returns
        -------
        GeneratedData
            Generated dataset with feature array X of shape
            (n_samples, n_features) and cluster labels y.
        """
        rng = self._rng

        # Generate or use provided centers
        if self.centers is None:
            centers = rng.uniform(
                self.center_box[0],
                self.center_box[1],
                (self.n_clusters, self.n_features)
            )
        else:
            centers = np.array(self.centers)
            self.n_clusters = len(centers)

        # Handle cluster_std
        if isinstance(self.cluster_std, (int, float)):
            stds = [self.cluster_std] * self.n_clusters
        else:
            stds = list(self.cluster_std)
            while len(stds) < self.n_clusters:
                stds.append(stds[-1])

        # Distribute samples among clusters
        n_samples_per_cluster = [self.n_samples // self.n_clusters] * self.n_clusters
        for i in range(self.n_samples % self.n_clusters):
            n_samples_per_cluster[i] += 1

        # Generate samples
        X_list = []
        y_list = []

        for i, (center, std, n) in enumerate(zip(centers, stds, n_samples_per_cluster)):
            X_cluster = rng.normal(center, std, (n, self.n_features))
            X_list.append(X_cluster)
            y_list.append(np.full(n, i, dtype=int))

        X = np.vstack(X_list)
        y = np.hstack(y_list)

        # Shuffle
        if self.shuffle:
            indices = rng.permutation(len(X))
            X = X[indices]
            y = y[indices]

        return GeneratedData(
            X=X,
            y=y,
            feature_names=[f"x{i}" for i in range(self.n_features)],
            target_names=[f"cluster{i}" for i in range(self.n_clusters)]
        )

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        schema = super().get_parameter_schema()
        schema["cluster_std"] = {
            "type": "number",
            "default": 1.0,
            "minimum": 0.0,
            "description": "Standard deviation of clusters"
        }
        schema["center_box"] = {
            "type": "array",
            "default": [-10.0, 10.0],
            "description": "Bounding box for centers"
        }
        return schema
