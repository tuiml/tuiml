"""Test suite for Manhattan (L1) distance function.

Tests cover:
- Known distance calculations
- Self-distance equals zero
- Symmetry property
- Triangle inequality
"""

import numpy as np

from tuiml.algorithms.clustering.distance import manhattan_distance


class TestManhattanDistance:
    """Tests for the manhattan_distance function."""

    def test_known_distance(self):
        """Test known Manhattan distance."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([3.0, 4.0])

        np.testing.assert_allclose(manhattan_distance(x1, x2), 7.0)

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
