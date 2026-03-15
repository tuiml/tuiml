"""Unsupervised algorithms for grouping similar data instances."""

# Base classes (from algorithms/base - single source of truth)
from tuiml.base.algorithms import (
    Clusterer,
    DensityBasedClusterer,
    UpdateableClusterer,
    clusterer,
)

# Distance functions
from tuiml.algorithms.clustering.distance import (
    euclidean_distance,
    manhattan_distance,
    cosine_distance,
    chebyshev_distance,
    minkowski_distance,
    pairwise_distances,
    get_distance_function,
)

# Clustering algorithms
from tuiml.algorithms.clustering.simple_kmeans import KMeansClusterer
from tuiml.algorithms.clustering.farthest_first import FarthestFirstClusterer
from tuiml.algorithms.clustering.hierarchical import AgglomerativeClusterer
from tuiml.algorithms.clustering.dbscan import DBSCANClusterer
from tuiml.algorithms.clustering.em import GaussianMixtureClusterer
from tuiml.algorithms.clustering.canopy import CanopyClusterer
from tuiml.algorithms.clustering.cobweb import CobwebClusterer
from tuiml.algorithms.clustering.filtered_clusterer import FilteredClusterer

__all__ = [
    "Clusterer",
    "DensityBasedClusterer",
    "UpdateableClusterer",
    "clusterer",
    "euclidean_distance",
    "manhattan_distance",
    "cosine_distance",
    "chebyshev_distance",
    "minkowski_distance",
    "pairwise_distances",
    "get_distance_function",
    "KMeansClusterer",
    "FarthestFirstClusterer",
    "AgglomerativeClusterer",
    "DBSCANClusterer",
    "GaussianMixtureClusterer",
    "CanopyClusterer",
    "CobwebClusterer",
    "FilteredClusterer",
]
