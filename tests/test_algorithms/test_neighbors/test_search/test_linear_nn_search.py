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
