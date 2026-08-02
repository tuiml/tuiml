"""Test suite for KDTree nearest neighbor search.

Tests cover:
- Building from data
- Query returns correct k neighbors
- query_radius finds points within radius
- Known nearest neighbor on simple data
- Edge cases
"""

import numpy as np
import pytest

from tuiml.algorithms.neighbors.search import KDTree


class TestKDTreeInit:
    """Tests for KDTree initialization."""

    def test_default_initialization(self):
        """Test default leaf_size initialization."""
        tree = KDTree()

        assert tree.leaf_size == 10
        assert tree._is_built is False

    def test_custom_leaf_size(self):
        """Test custom leaf_size initialization."""
        tree = KDTree(leaf_size=5)

        assert tree.leaf_size == 5


class TestKDTreeBuild:
    """Tests for the build() method."""

    def test_build_basic(self):
        """Test building a KDTree from data."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        tree = KDTree(leaf_size=10)

        result = tree.build(X)

        assert result is tree
        assert tree._is_built is True
        assert tree.n_samples_ == 50
        assert tree.n_features_ == 3
        assert tree._root is not None

    def test_build_small_data(self):
        """Test building with very small dataset (fewer points than leaf_size)."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        tree = KDTree(leaf_size=10)
        tree.build(X)

        assert tree._is_built is True
        assert tree.n_samples_ == 2

    def test_build_single_point(self):
        """Test building with a single data point."""
        X = np.array([[1.0, 2.0, 3.0]])
        tree = KDTree(leaf_size=10)
        tree.build(X)

        assert tree._is_built is True
        assert tree.n_samples_ == 1

    def test_repr_built(self):
        """Test string representation after building."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        tree = KDTree(leaf_size=2)
        tree.build(X)

        repr_str = repr(tree)
        assert "KDTree" in repr_str
        assert "n_samples=4" in repr_str


class TestKDTreeQuery:
    """Tests for the query() method."""

    def test_query_radius_zero_returns_exact_match(self):
        """Test that radius=0 returns only exact matches."""
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        tree = KDTree(leaf_size=2)
        tree.build(X)

        dists, indices = tree.query_radius(np.array([0.0, 0.0]), radius=0.0)

        assert len(indices) == 1
        assert indices[0] == 0
