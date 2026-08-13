"""Nearest neighbor search algorithms.

This module provides efficient data structures and algorithms for 
k-nearest neighbor and radius-based neighbor search.

Algorithms
----------
- **BruteForceSearch:** Simple brute force search.
- **KDTree:** Space-partitioning tree for low-dimensional data.
- **BallTree:** Hypersphere-partitioning tree for higher-dimensional data.
"""

from tuiml.base.neighbors import NearestNeighborSearch
from tuiml.algorithms.neighbors.search.brute_force import BruteForceSearch
from tuiml.algorithms.neighbors.search.kd_tree import KDTree
from tuiml.algorithms.neighbors.search.ball_tree import BallTree

__all__ = [
    "NearestNeighborSearch",
    "BruteForceSearch",
    "KDTree",
    "BallTree",
]
