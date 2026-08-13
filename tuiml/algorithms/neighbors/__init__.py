"""Neighbor-based (instance-based) algorithms.

This module provides classifiers and regressors that make predictions
based on similarity to training instances.

Algorithms
----------
- **KNearestNeighborsClassifier:** k-Nearest Neighbors classifier.
"""

from tuiml.algorithms.neighbors.knn import KNearestNeighborsClassifier, KNearestNeighborsRegressor
from tuiml.algorithms.neighbors.search import (
    NearestNeighborSearch,
    BruteForceSearch,
    KDTree,
    BallTree,
)

__all__ = [
    "KNearestNeighborsClassifier",
    "KNearestNeighborsRegressor",
    "NearestNeighborSearch",
    "BruteForceSearch",
    "KDTree",
    "BallTree",
]
