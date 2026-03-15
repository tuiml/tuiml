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

    def test_query_returns_correct_k(self):
        """Test that query returns exactly k neighbors."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        tree = BallTree(leaf_size=5)
        tree.build(X)

        k = 5
        dists, indices = tree.query(X[0], k=k)

        assert len(dists) == k
        assert len(indices) == k

    def test_query_nearest_is_self(self):
        """Test that the nearest neighbor of a training point is itself."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        tree = BallTree(leaf_size=2)
        tree.build(X)

        dists, indices = tree.query(X[0], k=1)

        assert indices[0] == 0
        np.testing.assert_allclose(dists[0], 0.0, atol=1e-10)

    def test_query_known_nearest_neighbor(self):
        """Test known nearest neighbor on simple data."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
        tree = BallTree(leaf_size=2)
        tree.build(X)

        # Query near (1, 0) -- closest should be index 1
        dists, indices = tree.query(np.array([0.9, 0.1]), k=1)

        assert indices[0] == 1

    def test_query_distances_sorted(self):
        """Test that returned distances are in ascending order."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        tree = BallTree(leaf_size=10)
        tree.build(X)

        dists, indices = tree.query(np.array([0.0, 0.0, 0.0]), k=10)

        for i in range(len(dists) - 1):
            assert dists[i] <= dists[i + 1] + 1e-10

    def test_query_k_exceeds_n_samples(self):
        """Test querying with k larger than number of samples."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        tree = BallTree(leaf_size=2)
        tree.build(X)

        dists, indices = tree.query(X[0], k=100)

        assert len(dists) == 3

    def test_query_before_build_raises(self):
        """Test that querying before build raises an error."""
        tree = BallTree()

        with pytest.raises(Exception):
            tree.query(np.array([1.0, 2.0]), k=1)


class TestBallTreeQueryRadius:
    """Tests for the query_radius() method."""

    def test_query_radius_finds_close_points(self):
        """Test that query_radius finds all points within the radius."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
        tree = BallTree(leaf_size=2)
        tree.build(X)

        dists, indices = tree.query_radius(np.array([0.0, 0.0]), radius=1.5)

        assert len(indices) == 3
        assert 0 in indices
        assert 1 in indices
        assert 2 in indices

    def test_query_radius_excludes_far_points(self):
        """Test that query_radius excludes points outside the radius."""
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        tree = BallTree(leaf_size=2)
        tree.build(X)

        dists, indices = tree.query_radius(np.array([0.0, 0.0]), radius=1.0)

        assert len(indices) == 1
        assert 0 in indices

    def test_query_radius_large_returns_all(self):
        """Test that a very large radius returns all points."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        tree = BallTree(leaf_size=5)
        tree.build(X)

        dists, indices = tree.query_radius(np.array([0.0, 0.0]), radius=1e6)

        assert len(indices) == 20

    def test_query_radius_empty_result(self):
        """Test query_radius with no points in range."""
        X = np.array([[100.0, 100.0], [200.0, 200.0]])
        tree = BallTree(leaf_size=2)
        tree.build(X)

        dists, indices = tree.query_radius(np.array([0.0, 0.0]), radius=1.0)

        assert len(indices) == 0


class TestBallTreeConsistency:
    """Tests comparing BallTree results with brute force."""

    def test_matches_brute_force(self):
        """Test that BallTree query matches brute force for known data."""
        np.random.seed(42)
        X = np.random.randn(30, 3)
        query = np.array([0.0, 0.0, 0.0])

        tree = BallTree(leaf_size=5)
        tree.build(X)
        bt_dists, bt_indices = tree.query(query, k=5)

        # Brute force
        dists = np.sqrt(np.sum((X - query) ** 2, axis=1))
        bf_indices = np.argsort(dists)[:5]
        bf_dists = dists[bf_indices]

        np.testing.assert_allclose(bt_dists, bf_dists, atol=1e-10)
        np.testing.assert_array_equal(bt_indices, bf_indices)
