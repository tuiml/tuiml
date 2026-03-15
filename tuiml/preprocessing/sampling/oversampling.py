"""
Random oversampling methods for imbalanced learning.

Simple oversampling techniques that duplicate minority samples.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict
from tuiml.base.preprocessing import Transformer

class RandomOverSampler(Transformer):
    """
    Random over-sampling by duplicating minority samples.

    Simply duplicates random minority class samples to balance classes.

    Parameters
    ----------
    sampling_strategy : float or str or dict, default='auto'
        Sampling strategy:
        - 'auto': balance all classes to match majority
        - 'minority': only oversample minority class
        - dict: {class_label: target_count}
    random_state : int, optional
        Random seed.
    shrinkage : float, optional
        If not None, adds Gaussian noise with this shrinkage factor.

    Examples
    --------
    >>> from tuiml.preprocessing.sampling import RandomOverSampler
    >>> ros = RandomOverSampler(sampling_strategy='auto')
    >>> X_res, y_res = ros.fit_resample(X, y)
    """

    def __init__(
        self,
        sampling_strategy: Union[float, str, dict] = 'auto',
        random_state: Optional[int] = None,
        shrinkage: Optional[float] = None
    ):
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.shrinkage = shrinkage
        self.sampling_strategy_: Dict = {}

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "sampling_strategy": {
                "type": ["string", "number", "object"],
                "default": "auto",
                "description": "Sampling strategy: 'auto', 'minority', or dict {class: target_count}"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility"
            },
            "shrinkage": {
                "type": ["number", "null"],
                "default": None,
                "description": "If set, adds Gaussian noise with this shrinkage factor"
            }
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomOverSampler":
        """Fit the sampler."""
        X, y = np.asarray(X), np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(classes, counts))
        max_count = counts.max()

        if isinstance(self.sampling_strategy, dict):
            self.sampling_strategy_ = self.sampling_strategy
        elif self.sampling_strategy in ['auto', 'not majority']:
            majority = classes[np.argmax(counts)]
            self.sampling_strategy_ = {
                c: max_count - n for c, n in class_counts.items()
                if c != majority and n < max_count
            }
        elif self.sampling_strategy == 'minority':
            minority = classes[np.argmin(counts)]
            self.sampling_strategy_ = {minority: max_count - class_counts[minority]}
        else:
            self.sampling_strategy_ = {}

        self._is_fitted = True
        return self

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and resample."""
        self.fit(X, y)
        rng = np.random.RandomState(self.random_state)

        X_res, y_res = [X.copy()], [y.copy()]

        for target_class, n_samples in self.sampling_strategy_.items():
            if n_samples <= 0:
                continue

            X_minority = X[y == target_class]
            indices = rng.choice(len(X_minority), size=n_samples, replace=True)
            X_new = X_minority[indices].copy()

            # Add noise if shrinkage is specified
            if self.shrinkage is not None:
                std = X_minority.std(axis=0) * self.shrinkage
                noise = rng.normal(0, std, X_new.shape)
                X_new += noise

            X_res.append(X_new)
            y_res.append(np.full(n_samples, target_class))

        return np.vstack(X_res), np.hstack(y_res)

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use fit_resample() instead")

    def __repr__(self) -> str:
        return f"RandomOverSampler(sampling_strategy={self.sampling_strategy!r})"

class ClusterOverSampler(Transformer):
    """
    Cluster-based oversampling.

    Clusters each class and oversamples within clusters to preserve
    data distribution.

    Parameters
    ----------
    sampling_strategy : str or dict, default='auto'
        Sampling strategy.
    n_clusters : int, default=5
        Number of clusters per class.
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> from tuiml.preprocessing.sampling import ClusterOverSampler
    >>> cbos = ClusterOverSampler(n_clusters=3)
    >>> X_res, y_res = cbos.fit_resample(X, y)
    """

    def __init__(
        self,
        sampling_strategy: Union[float, str, dict] = 'auto',
        n_clusters: int = 5,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.sampling_strategy_: Dict = {}

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "sampling_strategy": {
                "type": ["string", "number", "object"],
                "default": "auto",
                "description": "Sampling strategy"
            },
            "n_clusters": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "description": "Number of clusters per class"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility"
            }
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClusterOverSampler":
        """Fit the sampler."""
        X, y = np.asarray(X), np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(classes, counts))
        max_count = counts.max()

        if isinstance(self.sampling_strategy, dict):
            self.sampling_strategy_ = self.sampling_strategy
        else:
            majority = classes[np.argmax(counts)]
            self.sampling_strategy_ = {
                c: max_count - n for c, n in class_counts.items()
                if c != majority and n < max_count
            }

        self._is_fitted = True
        return self

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and resample."""
        self.fit(X, y)
        rng = np.random.RandomState(self.random_state)

        X_res, y_res = [X.copy()], [y.copy()]

        for target_class, n_samples in self.sampling_strategy_.items():
            if n_samples <= 0:
                continue

            X_minority = X[y == target_class]
            n_clusters = min(self.n_clusters, len(X_minority))

            # Simple k-means
            clusters = self._kmeans(X_minority, n_clusters, rng)

            # Sample proportionally from each cluster
            cluster_sizes = np.bincount(clusters, minlength=n_clusters)
            cluster_ratios = cluster_sizes / cluster_sizes.sum()
            samples_per_cluster = (cluster_ratios * n_samples).astype(int)

            # Adjust for rounding
            diff = n_samples - samples_per_cluster.sum()
            if diff > 0:
                samples_per_cluster[np.argmax(cluster_sizes)] += diff

            X_new_list = []
            for c in range(n_clusters):
                cluster_mask = clusters == c
                X_cluster = X_minority[cluster_mask]
                if len(X_cluster) > 0 and samples_per_cluster[c] > 0:
                    idx = rng.choice(len(X_cluster), samples_per_cluster[c], replace=True)
                    X_new_list.append(X_cluster[idx])

            if X_new_list:
                X_res.append(np.vstack(X_new_list))
                y_res.append(np.full(sum(len(x) for x in X_new_list), target_class))

        return np.vstack(X_res), np.hstack(y_res)

    def _kmeans(self, X: np.ndarray, k: int, rng: np.random.RandomState,
                max_iter: int = 50) -> np.ndarray:
        """Simple k-means."""
        n = X.shape[0]
        centroids = X[rng.choice(n, k, replace=False)]

        for _ in range(max_iter):
            dists = np.array([[np.sqrt(np.sum((x - c) ** 2)) for c in centroids] for x in X])
            clusters = np.argmin(dists, axis=1)
            new_centroids = np.array([
                X[clusters == i].mean(axis=0) if (clusters == i).any() else centroids[i]
                for i in range(k)
            ])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return clusters

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use fit_resample() instead")

    def __repr__(self) -> str:
        return f"ClusterOverSampler(n_clusters={self.n_clusters})"
