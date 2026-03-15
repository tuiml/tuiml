"""Nearest neighbor search algorithms.

This module provides efficient data structures and algorithms for 
k-nearest neighbor and radius-based neighbor search.

Algorithms
----------
- **LinearNNSearch:** Simple brute force search.
- **KDTree:** Space-partitioning tree for low-dimensional data.
- **BallTree:** Hypersphere-partitioning tree for higher-dimensional data.
"""

from tuiml.base.neighbors import NearestNeighborSearch
from tuiml.algorithms.neighbors.search.linear_nn_search import LinearNNSearch
from tuiml.algorithms.neighbors.search.kd_tree import KDTree
from tuiml.algorithms.neighbors.search.ball_tree import BallTree

__all__ = [
    "NearestNeighborSearch",
    "LinearNNSearch",
    "KDTree",
    "BallTree",
]
