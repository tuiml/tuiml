"""Test suite for Manhattan (L1) distance function.

Tests cover:
- Known distance calculations
- Self-distance equals zero
- Symmetry property
- Triangle inequality
"""

import numpy as np
import pytest

from tuiml.algorithms.clustering.distance import manhattan_distance


class TestManhattanDistance:
    """Tests for the manhattan_distance function."""

    def test_known_distance(self):
        """Test known Manhattan distance."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([3.0, 4.0])

        np.testing.assert_allclose(manhattan_distance(x1, x2), 7.0)

    def test_self_distance_is_zero(self):
        """Test that the distance from a point to itself is zero."""
        x = np.array([1.0, 2.0, 3.0])

        np.testing.assert_allclose(manhattan_distance(x, x), 0.0, atol=1e-15)

    def test_symmetry(self):
        """Test that d(x, y) == d(y, x)."""
        np.random.seed(42)
        x1 = np.random.randn(5)
        x2 = np.random.randn(5)

        np.testing.assert_allclose(
            manhattan_distance(x1, x2),
            manhattan_distance(x2, x1),
            atol=1e-15,
        )

    def test_triangle_inequality(self):
        """Test the triangle inequality: d(x,z) <= d(x,y) + d(y,z)."""
        np.random.seed(42)
        x = np.random.randn(4)
        y = np.random.randn(4)
        z = np.random.randn(4)

        d_xz = manhattan_distance(x, z)
        d_xy = manhattan_distance(x, y)
        d_yz = manhattan_distance(y, z)

        assert d_xz <= d_xy + d_yz + 1e-10

    def test_unit_vectors(self):
        """Test distance between unit vectors along axes."""
        x1 = np.array([1.0, 0.0, 0.0])
        x2 = np.array([0.0, 1.0, 0.0])

        # |1-0| + |0-1| + |0-0| = 2.0
        np.testing.assert_allclose(manhattan_distance(x1, x2), 2.0)

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        x1 = np.array([-1.0, -2.0])
        x2 = np.array([2.0, 2.0])

        # |(-1)-2| + |(-2)-2| = 3 + 4 = 7
        np.testing.assert_allclose(manhattan_distance(x1, x2), 7.0)

    def test_single_dimension(self):
        """Test in a single dimension."""
        x1 = np.array([3.0])
        x2 = np.array([7.0])

        np.testing.assert_allclose(manhattan_distance(x1, x2), 4.0)

    def test_non_negative(self):
        """Test that Manhattan distance is always non-negative."""
        np.random.seed(42)
        for _ in range(20):
            x1 = np.random.randn(5)
            x2 = np.random.randn(5)
            assert manhattan_distance(x1, x2) >= 0.0
