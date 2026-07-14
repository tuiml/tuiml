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

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

if SKLEARN_AVAILABLE:
    from tuiml.algorithms.neighbors.sklearn_knn_clf import SklearnKNeighborsClassifier
    from tuiml.algorithms.neighbors.sklearn_knn_reg import SklearnKNeighborsRegressor

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

if SKLEARN_AVAILABLE:
    __all__.extend([
        "SklearnKNeighborsClassifier",
        "SklearnKNeighborsRegressor",
    ])

