"""Base class for nearest neighbor search algorithms.

Defines the common interface (``build``/``query``) shared by search
strategies such as brute force, KD-tree, and Ball-tree.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List

class NearestNeighborSearch(ABC):
    """Abstract base class for nearest neighbor search algorithms.

    Provides a common interface for different search strategies like
    brute force, KD-tree, Ball-tree, etc.

    Attributes
    ----------
    X_ : np.ndarray of shape (n_samples, n_features)
        Training data (set by ``build()``).
    n_samples_ : int
        Number of training samples.
    n_features_ : int
        Number of features.
    """

    def __init__(self):
        """Initialize the search algorithm."""
        self._is_built = False
        self.X_ = None
        self.n_samples_ = None
        self.n_features_ = None

    @abstractmethod
    def build(self, X: np.ndarray) -> "NearestNeighborSearch":
        """Build the search structure from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : NearestNeighborSearch
            The built search structure, for method chaining.
        """
        pass

    @abstractmethod
    def query(self, x: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Find the ``k`` nearest neighbors of a single query point.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            Query point.
        k : int, default=1
            Number of neighbors to find.

        Returns
        -------
        distances : np.ndarray of shape (k,)
            Distance to each neighbor.
        indices : np.ndarray of shape (k,)
            Index of each neighbor in the training data.
        """
        pass

    def query_batch(self, X: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Find the ``k`` nearest neighbors of multiple query points.

        Parameters
        ----------
        X : np.ndarray of shape (n_queries, n_features)
            Query points.
        k : int, default=1
            Number of neighbors to find.

        Returns
        -------
        distances : np.ndarray of shape (n_queries, k)
            Distance to each neighbor for each query.
        indices : np.ndarray of shape (n_queries, k)
            Index of each neighbor in the training data.
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
        """Find all neighbors within a given radius.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            Query point.
        radius : float
            Maximum distance.

        Returns
        -------
        distances : np.ndarray
            Distances to the neighbors within ``radius``.
        indices : np.ndarray
            Indices of the neighbors within ``radius``.
        """
        # Default implementation using brute force
        distances, indices = self.query(x, self.n_samples_)
        mask = distances <= radius
        return distances[mask], indices[mask]

    @staticmethod
    def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute the Euclidean distance between two points.

        Parameters
        ----------
        x1 : np.ndarray of shape (n_features,)
            First point.
        x2 : np.ndarray of shape (n_features,)
            Second point.

        Returns
        -------
        distance : float
            Euclidean distance :math:`\\sqrt{\\sum_i (x_{1i} - x_{2i})^2}`.
        """
        diff = x1 - x2
        return np.sqrt(np.sum(diff ** 2))

    @staticmethod
    def euclidean_distance_squared(x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute the squared Euclidean distance (faster, avoids the square root).

        Parameters
        ----------
        x1 : np.ndarray of shape (n_features,)
            First point.
        x2 : np.ndarray of shape (n_features,)
            Second point.

        Returns
        -------
        distance : float
            Squared Euclidean distance between the points.
        """
        diff = x1 - x2
        return np.sum(diff ** 2)

    def _check_is_built(self):
        """Raise ``RuntimeError`` if the search structure has not been built."""
        if not self._is_built:
            raise RuntimeError("Search structure not built. Call build() first.")

    def __repr__(self) -> str:
        """Return string representation of the search structure."""
        name = self.__class__.__name__
        if self._is_built:
            return f"{name}(n_samples={self.n_samples_}, n_features={self.n_features_})"
        return f"{name}(not built)"
