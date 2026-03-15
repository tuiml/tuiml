"""Test suite for LinearNNSearch (brute force) nearest neighbor search.

Tests cover:
- Building from data
- Query returns correct k neighbors
- query_radius finds points within radius
- Known nearest neighbor on simple data
- Consistency with manual computation
"""

import numpy as np
import pytest

from tuiml.algorithms.neighbors.search import LinearNNSearch


class TestLinearNNSearchInit:
    """Tests for LinearNNSearch initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        search = LinearNNSearch()

        assert search._is_built is False


class TestLinearNNSearchBuild:
    """Tests for the build() method."""

    def test_build_basic(self):
        """Test building from data."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        search = LinearNNSearch()

        result = search.build(X)

        assert result is search
        assert search._is_built is True
        assert search.n_samples_ == 50
        assert search.n_features_ == 3

    def test_build_small_data(self):
        """Test building with very small dataset."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        search = LinearNNSearch()
        search.build(X)

        assert search._is_built is True
        assert search.n_samples_ == 2

    def test_build_single_point(self):
        """Test building with a single data point."""
        X = np.array([[1.0, 2.0, 3.0]])
        search = LinearNNSearch()
        search.build(X)

        assert search._is_built is True
        assert search.n_samples_ == 1

    def test_repr_built(self):
        """Test string representation after building."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        search = LinearNNSearch()
        search.build(X)

        repr_str = repr(search)
        assert "LinearNNSearch" in repr_str
        assert "n_samples=3" in repr_str


class TestLinearNNSearchQuery:
    """Tests for the query() method."""

    def test_query_returns_correct_k(self):
        """Test that query returns exactly k neighbors."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        search = LinearNNSearch()
        search.build(X)

        k = 5
        dists, indices = search.query(X[0], k=k)

        assert len(dists) == k
        assert len(indices) == k

    def test_query_nearest_is_self(self):
        """Test that the nearest neighbor of a training point is itself."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        search = LinearNNSearch()
        search.build(X)

        dists, indices = search.query(X[0], k=1)

        assert indices[0] == 0
        np.testing.assert_allclose(dists[0], 0.0, atol=1e-10)

    def test_query_known_nearest_neighbor(self):
        """Test known nearest neighbor on simple data."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
        search = LinearNNSearch()
        search.build(X)

        dists, indices = search.query(np.array([0.9, 0.1]), k=1)

        assert indices[0] == 1

    def test_query_distances_sorted(self):
        """Test that returned distances are in ascending order."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        search = LinearNNSearch()
        search.build(X)

        dists, indices = search.query(np.array([0.0, 0.0, 0.0]), k=10)

        for i in range(len(dists) - 1):
            assert dists[i] <= dists[i + 1] + 1e-10

    def test_query_k_exceeds_n_samples(self):
        """Test querying with k larger than number of samples."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        search = LinearNNSearch()
        search.build(X)

        dists, indices = search.query(X[0], k=100)

        assert len(dists) == 3

    def test_query_before_build_raises(self):
        """Test that querying before build raises an error."""
        search = LinearNNSearch()

        with pytest.raises(Exception):
            search.query(np.array([1.0, 2.0]), k=1)

    def test_query_all_distances_correct(self):
        """Test that all returned distances match manual computation."""
        X = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
        search = LinearNNSearch()
        search.build(X)

        query = np.array([0.0, 0.0])
        dists, indices = search.query(query, k=3)

        # Distances: 0.0, 5.0, 1.0 -> sorted: 0.0, 1.0, 5.0
        np.testing.assert_allclose(dists[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(dists[1], 1.0, atol=1e-10)
        np.testing.assert_allclose(dists[2], 5.0, atol=1e-10)


class TestLinearNNSearchQueryRadius:
    """Tests for the query_radius() method."""

    def test_query_radius_finds_close_points(self):
        """Test that query_radius finds all points within the radius."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
        search = LinearNNSearch()
        search.build(X)

        dists, indices = search.query_radius(np.array([0.0, 0.0]), radius=1.5)

        assert len(indices) == 3
        assert 0 in indices
        assert 1 in indices
        assert 2 in indices

    def test_query_radius_excludes_far_points(self):
        """Test that query_radius excludes points outside the radius."""
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        search = LinearNNSearch()
        search.build(X)

        dists, indices = search.query_radius(np.array([0.0, 0.0]), radius=1.0)

        assert len(indices) == 1
        assert 0 in indices

    def test_query_radius_large_returns_all(self):
        """Test that a very large radius returns all points."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        search = LinearNNSearch()
        search.build(X)

        dists, indices = search.query_radius(np.array([0.0, 0.0]), radius=1e6)

        assert len(indices) == 20

    def test_query_radius_results_sorted(self):
        """Test that query_radius results are sorted by distance."""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        search = LinearNNSearch()
        search.build(X)

        dists, indices = search.query_radius(np.array([0.0, 0.0]), radius=2.0)

        if len(dists) > 1:
            for i in range(len(dists) - 1):
                assert dists[i] <= dists[i + 1] + 1e-10
