"""Unsupervised algorithms for grouping similar data instances.

Clusterers find structure without labels. They differ mainly in what they
assume a cluster *is* — a centroid, a density region, or a node in a
hierarchy — so the right choice depends on the shape you expect.

Algorithms
----------
- **KMeansClusterer:** Partition into ``k`` spherical clusters by centroid.
- **GaussianMixtureClusterer:** Soft assignment via expectation-maximisation.
- **DBSCANClusterer:** Density-based; finds arbitrary shapes and labels
  sparse points as noise, without being told how many clusters to expect.
- **DensityBasedClusterer:** Density estimation wrapped around a clusterer.
- **AgglomerativeClusterer:** Bottom-up hierarchy of merges.

Notes
-----
Every distance-based clusterer here takes its metric from
:mod:`~tuiml.algorithms.clustering.distance`, which is re-exported at this
level. Scale features first when using Euclidean distance, or the
widest-ranging column silently dominates the metric.

Examples
--------
>>> from tuiml.algorithms.clustering import KMeansClusterer
>>> from tuiml.datasets import load_iris
>>> data = load_iris()
>>> labels = KMeansClusterer(n_clusters=3, random_state=0).fit_predict(data.X)
>>> len(set(labels.tolist()))
3
"""

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
from tuiml.algorithms.clustering.kmeans import KMeansClusterer
from tuiml.algorithms.clustering.hierarchical import AgglomerativeClusterer
from tuiml.algorithms.clustering.dbscan import DBSCANClusterer
from tuiml.algorithms.clustering.gaussian_mixture import GaussianMixtureClusterer

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
    "AgglomerativeClusterer",
    "DBSCANClusterer",
    "GaussianMixtureClusterer",
]
