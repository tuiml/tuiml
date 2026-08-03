"""Distance functions used by clustering and neighbour search.

Merged from: test_distance_euclidean.py, test_distance_manhattan.py, test_distance_chebyshev.py, test_distance_minkowski.py, test_distance_cosine.py
"""

import numpy as np
from tuiml.algorithms.clustering.distance import euclidean_distance, euclidean_pairwise
from tuiml.algorithms.clustering.distance import manhattan_distance
from tuiml.algorithms.clustering.distance import chebyshev_distance
from tuiml.algorithms.clustering.distance import (
    minkowski_distance,
    euclidean_distance,
    manhattan_distance,
)
from tuiml.algorithms.clustering.distance import cosine_distance


# --------------------------------------------------------------------------
# Test suite for Euclidean distance functions.
# --------------------------------------------------------------------------

class TestEuclideanDistance:
    """Tests for the euclidean_distance function."""

    def test_known_distance_3_4(self):
        """Test known 3-4-5 right triangle distance."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([3.0, 4.0])

        np.testing.assert_allclose(euclidean_distance(x1, x2), 5.0)

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
                    D[i, j], euclidean_distance(X[i], X[j]), atol=1e-7
                )


# --------------------------------------------------------------------------
# Test suite for Manhattan (L1) distance function.
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Test suite for Chebyshev (L-infinity) distance function.
# --------------------------------------------------------------------------

class TestChebyshevDistance:
    """Tests for the chebyshev_distance function."""

    def test_known_distance(self):
        """Test known Chebyshev distance: max(|3-0|, |4-0|) = 4."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([3.0, 4.0])

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

    def test_less_than_or_equal_manhattan(self):
        """Test that Chebyshev distance <= Manhattan distance."""
        np.random.seed(42)
        from tuiml.algorithms.clustering.distance import manhattan_distance

        for _ in range(20):
            x1 = np.random.randn(5)
            x2 = np.random.randn(5)
            assert chebyshev_distance(x1, x2) <= manhattan_distance(x1, x2) + 1e-10


# --------------------------------------------------------------------------
# Test suite for Minkowski distance function.
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Test suite for cosine distance function.
# --------------------------------------------------------------------------

class TestCosineDistance:
    """Tests for the cosine_distance function."""

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

    def test_magnitude_invariance(self):
        """Test that cosine distance is invariant to scaling."""
        x1 = np.array([1.0, 2.0])
        x2 = np.array([3.0, 1.0])

        d_original = cosine_distance(x1, x2)
        d_scaled = cosine_distance(x1 * 100.0, x2 * 0.01)

        np.testing.assert_allclose(d_original, d_scaled, atol=1e-10)
