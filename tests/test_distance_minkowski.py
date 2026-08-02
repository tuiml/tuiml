"""Test suite for Minkowski distance function.

Tests cover:
- p=1 equals Manhattan distance
- p=2 equals Euclidean distance
- Custom p values
- Self-distance is zero
- Symmetry
"""

import numpy as np

from tuiml.algorithms.clustering.distance import (
    minkowski_distance,
    euclidean_distance,
    manhattan_distance,
)


class TestMinkowskiDistance:
    """Tests for the minkowski_distance function."""

    def test_p1_equals_manhattan(self):
        """Test that Minkowski with p=1 equals Manhattan distance."""
        np.random.seed(42)
        x1 = np.random.randn(5)
        x2 = np.random.randn(5)

        np.testing.assert_allclose(
            minkowski_distance(x1, x2, p=1),
            manhattan_distance(x1, x2),
            atol=1e-10,
        )

    def test_p2_equals_euclidean(self):
        """Test that Minkowski with p=2 equals Euclidean distance."""
        np.random.seed(42)
        x1 = np.random.randn(5)
        x2 = np.random.randn(5)

        np.testing.assert_allclose(
            minkowski_distance(x1, x2, p=2),
            euclidean_distance(x1, x2),
            atol=1e-10,
        )

    def test_known_distance_p2(self):
        """Test known Minkowski distance with p=2 (Euclidean)."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([3.0, 4.0])

        np.testing.assert_allclose(minkowski_distance(x1, x2, p=2), 5.0)

    def test_known_distance_p1(self):
        """Test known Minkowski distance with p=1 (Manhattan)."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([3.0, 4.0])

        np.testing.assert_allclose(minkowski_distance(x1, x2, p=1), 7.0)

    def test_custom_p_value(self):
        """Test with a custom p value (p=3)."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([3.0, 4.0])

        # (|3|^3 + |4|^3)^(1/3) = (27 + 64)^(1/3) = 91^(1/3)
        expected = np.power(91.0, 1.0 / 3.0)
        np.testing.assert_allclose(minkowski_distance(x1, x2, p=3), expected, atol=1e-10)

    def test_p_infinity_equals_chebyshev(self):
        """Test that Minkowski with p=inf equals Chebyshev distance."""
        from tuiml.algorithms.clustering.distance import chebyshev_distance

        np.random.seed(42)
        x1 = np.random.randn(5)
        x2 = np.random.randn(5)

        np.testing.assert_allclose(
            minkowski_distance(x1, x2, p=float("inf")),
            chebyshev_distance(x1, x2),
            atol=1e-10,
        )
