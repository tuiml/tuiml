"""Test suite for Euclidean distance functions.

Tests cover:
- Known distance calculations
- Self-distance equals zero
- Symmetry property
- Pairwise matrix shape and diagonal
- Higher dimensions
"""

import numpy as np
import pytest

from tuiml.algorithms.clustering.distance import euclidean_distance, euclidean_pairwise


class TestEuclideanDistance:
    """Tests for the euclidean_distance function."""

    def test_known_distance_3_4(self):
        """Test known 3-4-5 right triangle distance."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([3.0, 4.0])

        np.testing.assert_allclose(euclidean_distance(x1, x2), 5.0)

    def test_self_distance_is_zero(self):
        """Test that the distance from a point to itself is zero."""
        x = np.array([1.0, 2.0, 3.0])

        np.testing.assert_allclose(euclidean_distance(x, x), 0.0, atol=1e-15)

    def test_symmetry(self):
        """Test that d(x, y) == d(y, x)."""
        np.random.seed(42)
        x1 = np.random.randn(5)
        x2 = np.random.randn(5)

        np.testing.assert_allclose(
            euclidean_distance(x1, x2),
            euclidean_distance(x2, x1),
            atol=1e-15,
        )

    def test_unit_vectors(self):
        """Test distance between unit vectors along axes."""
        x1 = np.array([1.0, 0.0])
        x2 = np.array([0.0, 1.0])

        np.testing.assert_allclose(euclidean_distance(x1, x2), np.sqrt(2.0))

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        x1 = np.array([-1.0, -2.0])
        x2 = np.array([2.0, 2.0])

        expected = np.sqrt(9.0 + 16.0)
        np.testing.assert_allclose(euclidean_distance(x1, x2), expected)

    def test_high_dimensional(self):
        """Test in high dimensions."""
        np.random.seed(42)
        x1 = np.zeros(100)
        x2 = np.ones(100)

        # Distance should be sqrt(100) = 10
        np.testing.assert_allclose(euclidean_distance(x1, x2), 10.0)


class TestEuclideanPairwise:
    """Tests for the euclidean_pairwise function."""

    def test_pairwise_shape(self):
        """Test that pairwise matrix has correct shape."""
        np.random.seed(42)
        X = np.random.randn(5, 3)

        D = euclidean_pairwise(X)

        assert D.shape == (5, 5)

    def test_pairwise_diagonal_is_zero(self):
        """Test that the diagonal of the pairwise matrix is approximately zero."""
        np.random.seed(42)
        X = np.random.randn(10, 4)

        D = euclidean_pairwise(X)

        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-7)

    def test_pairwise_symmetry(self):
        """Test that the pairwise distance matrix is symmetric."""
        np.random.seed(42)
        X = np.random.randn(8, 3)

        D = euclidean_pairwise(X)

        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_pairwise_non_negative(self):
        """Test that all pairwise distances are non-negative."""
        np.random.seed(42)
        X = np.random.randn(10, 5)

        D = euclidean_pairwise(X)

        assert np.all(D >= -1e-10)

    def test_pairwise_with_two_sets(self):
        """Test pairwise distances between two different sets."""
        X = np.array([[0.0, 0.0], [1.0, 0.0]])
        Y = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])

        D = euclidean_pairwise(X, Y)

        assert D.shape == (2, 3)
        np.testing.assert_allclose(D[0, 0], 1.0)  # (0,0) -> (0,1)
        np.testing.assert_allclose(D[1, 2], 1.0)  # (1,0) -> (2,0)

    def test_pairwise_consistent_with_pointwise(self):
        """Test that pairwise matrix matches individual distance calculations."""
        np.random.seed(42)
        X = np.random.randn(5, 3)

        D = euclidean_pairwise(X)

        for i in range(5):
            for j in range(5):
                np.testing.assert_allclose(
                    D[i, j], euclidean_distance(X[i], X[j]), atol=1e-10
                )
