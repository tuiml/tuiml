"""Neighbor-based (instance-based) algorithms.

This module provides classifiers and regressors that make predictions
based on similarity to training instances.

Algorithms
----------
- **KNearestNeighborsClassifier:** k-Nearest Neighbors classifier.
- **KStarClassifier:** Entropy-based instance-based classifier.
- **LocallyWeightedLearningRegressor:** Locally Weighted Learning for regression.
"""

from tuiml.algorithms.neighbors.ibk import KNearestNeighborsClassifier, KNearestNeighborsRegressor
from tuiml.algorithms.neighbors.kstar import KStarClassifier
from tuiml.algorithms.neighbors.lwl import LocallyWeightedLearningRegressor
from tuiml.algorithms.neighbors.search import (
    NearestNeighborSearch,
    LinearNNSearch,
    KDTree,
    BallTree,
)

__all__ = [
    "KNearestNeighborsClassifier",
    "KNearestNeighborsRegressor",
    "KStarClassifier",
    "LocallyWeightedLearningRegressor",
    "NearestNeighborSearch",
    "LinearNNSearch",
    "KDTree",
    "BallTree",
]
