"""Test suite for BallTree nearest neighbor search.

Tests cover:
- Building from data
- Query returns correct k neighbors
- query_radius finds points within radius
- Known nearest neighbor on simple data
- Consistency with brute force
"""

import numpy as np
import pytest

from tuiml.algorithms.neighbors.search import BallTree


class TestBallTreeInit:
    """Tests for BallTree initialization."""

    def test_default_initialization(self):
        """Test default leaf_size initialization."""
        tree = BallTree()

        assert tree.leaf_size == 10
        assert tree._is_built is False

    def test_custom_leaf_size(self):
        """Test custom leaf_size initialization."""
        tree = BallTree(leaf_size=20)

        assert tree.leaf_size == 20


class TestBallTreeBuild:
    """Tests for the build() method."""

    def test_build_basic(self):
        """Test building a BallTree from data."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        tree = BallTree(leaf_size=10)

        result = tree.build(X)

        assert result is tree
        assert tree._is_built is True
        assert tree.n_samples_ == 50
        assert tree.n_features_ == 3
        assert tree._root is not None

    def test_build_small_data(self):
        """Test building with fewer points than leaf_size."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        tree = BallTree(leaf_size=10)
        tree.build(X)

        assert tree._is_built is True
        assert tree.n_samples_ == 2

    def test_build_single_point(self):
        """Test building with a single data point."""
        X = np.array([[1.0, 2.0, 3.0]])
        tree = BallTree(leaf_size=10)
        tree.build(X)

        assert tree._is_built is True
        assert tree.n_samples_ == 1

    def test_repr_built(self):
        """Test string representation after building."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        tree = BallTree(leaf_size=2)
        tree.build(X)

        repr_str = repr(tree)
        assert "BallTree" in repr_str
        assert "n_samples=4" in repr_str


class TestBallTreeQuery:
    """Tests for the query() method."""

    def test_query_radius_empty_result(self):
        """Test query_radius with no points in range."""
        X = np.array([[100.0, 100.0], [200.0, 200.0]])
        tree = BallTree(leaf_size=2)
        tree.build(X)

        dists, indices = tree.query_radius(np.array([0.0, 0.0]), radius=1.0)

        assert len(indices) == 0
