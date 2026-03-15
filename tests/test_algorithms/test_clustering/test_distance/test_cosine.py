"""Test suite for cosine distance function.

Tests cover:
- Identical vectors give distance 0
- Orthogonal vectors give distance 1
- Opposite vectors give distance 2
- Zero vector handling
"""

import numpy as np
import pytest

from tuiml.algorithms.clustering.distance import cosine_distance


class TestCosineDistance:
    """Tests for the cosine_distance function."""

    def test_identical_vectors_distance_zero(self):
        """Test that identical vectors have cosine distance 0."""
        x = np.array([1.0, 2.0, 3.0])

        np.testing.assert_allclose(cosine_distance(x, x), 0.0, atol=1e-10)

    def test_orthogonal_vectors_distance_one(self):
        """Test that orthogonal vectors have cosine distance 1."""
        x1 = np.array([1.0, 0.0])
        x2 = np.array([0.0, 1.0])

        np.testing.assert_allclose(cosine_distance(x1, x2), 1.0, atol=1e-10)

    def test_opposite_vectors_distance_two(self):
        """Test that opposite vectors have cosine distance 2."""
        x1 = np.array([1.0, 0.0])
        x2 = np.array([-1.0, 0.0])

        np.testing.assert_allclose(cosine_distance(x1, x2), 2.0, atol=1e-10)

    def test_parallel_same_direction(self):
        """Test that parallel same-direction vectors have distance 0."""
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([2.0, 4.0, 6.0])

        np.testing.assert_allclose(cosine_distance(x1, x2), 0.0, atol=1e-10)

    def test_zero_vector_returns_one(self):
        """Test that a zero vector returns distance 1."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([1.0, 2.0])

        assert cosine_distance(x1, x2) == 1.0

    def test_both_zero_vectors(self):
        """Test distance between two zero vectors."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([0.0, 0.0])

        assert cosine_distance(x1, x2) == 1.0

    def test_range_0_to_2(self):
        """Test that cosine distance is in the range [0, 2]."""
        np.random.seed(42)
        for _ in range(50):
            x1 = np.random.randn(5)
            x2 = np.random.randn(5)
            d = cosine_distance(x1, x2)
            assert -1e-10 <= d <= 2.0 + 1e-10

    def test_symmetry(self):
        """Test that d(x, y) == d(y, x)."""
        np.random.seed(42)
        x1 = np.random.randn(4)
        x2 = np.random.randn(4)

        np.testing.assert_allclose(
            cosine_distance(x1, x2),
            cosine_distance(x2, x1),
            atol=1e-10,
        )

    def test_magnitude_invariance(self):
        """Test that cosine distance is invariant to scaling."""
        x1 = np.array([1.0, 2.0])
        x2 = np.array([3.0, 1.0])

        d_original = cosine_distance(x1, x2)
        d_scaled = cosine_distance(x1 * 100.0, x2 * 0.01)

        np.testing.assert_allclose(d_original, d_scaled, atol=1e-10)
