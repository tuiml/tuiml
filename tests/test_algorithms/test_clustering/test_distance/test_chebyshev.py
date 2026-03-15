"""Test suite for Chebyshev (L-infinity) distance function.

Tests cover:
- Known distance (max absolute difference)
- Self-distance equals zero
- Symmetry property
- Various dimensions
"""

import numpy as np
import pytest

from tuiml.algorithms.clustering.distance import chebyshev_distance


class TestChebyshevDistance:
    """Tests for the chebyshev_distance function."""

    def test_known_distance(self):
        """Test known Chebyshev distance: max(|3-0|, |4-0|) = 4."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([3.0, 4.0])

        np.testing.assert_allclose(chebyshev_distance(x1, x2), 4.0)

    def test_self_distance_is_zero(self):
        """Test that the distance from a point to itself is zero."""
        x = np.array([1.0, 2.0, 3.0])

        np.testing.assert_allclose(chebyshev_distance(x, x), 0.0, atol=1e-15)

    def test_symmetry(self):
        """Test that d(x, y) == d(y, x)."""
        np.random.seed(42)
        x1 = np.random.randn(5)
        x2 = np.random.randn(5)

        np.testing.assert_allclose(
            chebyshev_distance(x1, x2),
            chebyshev_distance(x2, x1),
            atol=1e-15,
        )

    def test_single_dimension(self):
        """Test in a single dimension."""
        x1 = np.array([3.0])
        x2 = np.array([7.0])

        np.testing.assert_allclose(chebyshev_distance(x1, x2), 4.0)

    def test_max_on_specific_dimension(self):
        """Test that the max is correctly identified across dimensions."""
        x1 = np.array([0.0, 0.0, 0.0])
        x2 = np.array([1.0, 5.0, 2.0])

        # max(|1|, |5|, |2|) = 5
        np.testing.assert_allclose(chebyshev_distance(x1, x2), 5.0)

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        x1 = np.array([-3.0, 2.0])
        x2 = np.array([4.0, -1.0])

        # max(|(-3)-4|, |2-(-1)|) = max(7, 3) = 7
        np.testing.assert_allclose(chebyshev_distance(x1, x2), 7.0)

    def test_non_negative(self):
        """Test that Chebyshev distance is always non-negative."""
        np.random.seed(42)
        for _ in range(20):
            x1 = np.random.randn(5)
            x2 = np.random.randn(5)
            assert chebyshev_distance(x1, x2) >= 0.0

    def test_less_than_or_equal_manhattan(self):
        """Test that Chebyshev distance <= Manhattan distance."""
        np.random.seed(42)
        from tuiml.algorithms.clustering.distance import manhattan_distance

        for _ in range(20):
            x1 = np.random.randn(5)
            x2 = np.random.randn(5)
            assert chebyshev_distance(x1, x2) <= manhattan_distance(x1, x2) + 1e-10
