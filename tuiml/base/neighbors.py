"""
Base class for Nearest Neighbor Search algorithms.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List

class NearestNeighborSearch(ABC):
    """
    Abstract base class for nearest neighbor search algorithms.

    Provides a common interface for different search strategies like
    brute force, KD-tree, Ball-tree, etc.
    """

    def __init__(self):
        """Initialize the search algorithm."""
        self._is_built = False
        self.X_ = None
        self.n_samples_ = None
        self.n_features_ = None

    @abstractmethod
    def build(self, X: np.ndarray) -> "NearestNeighborSearch":
        """
        Build the search structure from training data.

        Args:
            X: Training data (n_samples, n_features)

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def query(self, x: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for a single query point.

        Args:
            x: Query point (n_features,)
            k: Number of neighbors to find

        Returns:
            Tuple of (distances, indices) where:
                - distances: Distance to each neighbor (k,)
                - indices: Index of each neighbor in training data (k,)
        """
        pass

    def query_batch(self, X: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for multiple query points.

        Args:
            X: Query points (n_queries, n_features)
            k: Number of neighbors to find

        Returns:
            Tuple of (distances, indices) where:
                - distances: Distance to each neighbor (n_queries, k)
                - indices: Index of each neighbor (n_queries, k)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_queries = X.shape[0]
        all_distances = np.zeros((n_queries, k))
        all_indices = np.zeros((n_queries, k), dtype=int)

        for i in range(n_queries):
            distances, indices = self.query(X[i], k)
            all_distances[i] = distances
            all_indices[i] = indices

        return all_distances, all_indices

    def query_radius(self, x: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find all neighbors within a given radius.

        Args:
            x: Query point (n_features,)
            radius: Maximum distance

        Returns:
            Tuple of (distances, indices) for neighbors within radius
        """
        # Default implementation using brute force
        distances, indices = self.query(x, self.n_samples_)
        mask = distances <= radius
        return distances[mask], indices[mask]

    @staticmethod
    def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Euclidean distance between two points."""
        diff = x1 - x2
        return np.sqrt(np.sum(diff ** 2))

    @staticmethod
    def euclidean_distance_squared(x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute squared Euclidean distance (faster, avoids sqrt)."""
        diff = x1 - x2
        return np.sum(diff ** 2)

    def _check_is_built(self):
        """Check if the search structure has been built."""
        if not self._is_built:
            raise RuntimeError("Search structure not built. Call build() first.")

    def __repr__(self) -> str:
        """String representation."""
        name = self.__class__.__name__
        if self._is_built:
            return f"{name}(n_samples={self.n_samples_}, n_features={self.n_features_})"
        return f"{name}(not built)"
