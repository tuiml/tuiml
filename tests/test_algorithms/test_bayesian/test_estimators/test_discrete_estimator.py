"""Test suite for DiscreteEstimator probability estimator.

Tests cover:
- Initialization with and without Laplace smoothing
- Uniform distribution after equal additions
- Laplace smoothing vs no smoothing behavior
- Edge cases (out of bounds, zero counts)
"""

import numpy as np
import pytest

from tuiml.algorithms.bayesian.estimators import DiscreteEstimator


class TestDiscreteEstimatorInit:
    """Tests for DiscreteEstimator initialization."""

    def test_default_initialization(self):
        """Test default initialization with Laplace smoothing enabled."""
        est = DiscreteEstimator(num_symbols=5)

        assert est.num_symbols == 5
        assert est.laplace is True
        assert est.total_count == 0.0
        np.testing.assert_array_equal(est.counts, np.zeros(5))

    def test_initialization_without_laplace(self):
        """Test initialization with Laplace smoothing disabled."""
        est = DiscreteEstimator(num_symbols=3, laplace=False)

        assert est.num_symbols == 3
        assert est.laplace is False

    def test_initialization_single_symbol(self):
        """Test initialization with a single symbol."""
        est = DiscreteEstimator(num_symbols=1)

        assert est.num_symbols == 1
        assert len(est.counts) == 1


class TestDiscreteEstimatorAddValue:
    """Tests for the add_value() method."""

    def test_add_value_increments_count(self):
        """Test that adding a value increments the correct symbol count."""
        est = DiscreteEstimator(num_symbols=3)
        est.add_value(0)
        est.add_value(0)
        est.add_value(1)

        assert est.get_count(0) == 2.0
        assert est.get_count(1) == 1.0
        assert est.get_count(2) == 0.0
        assert est.total_count == 3.0

    def test_add_value_out_of_bounds_ignored(self):
        """Test that out-of-bounds values are silently ignored."""
        est = DiscreteEstimator(num_symbols=3)
        est.add_value(-1)
        est.add_value(3)
        est.add_value(100)

        assert est.total_count == 0.0

    def test_add_weighted_value(self):
        """Test adding a weighted value."""
        est = DiscreteEstimator(num_symbols=3)
        est.add_value(1, weight=5.0)

        assert est.get_count(1) == 5.0
        assert est.total_count == 5.0


class TestDiscreteEstimatorProbability:
    """Tests for get_probability()."""

    def test_uniform_distribution_with_laplace(self):
        """Test uniform distribution after equal additions with Laplace smoothing."""
        est = DiscreteEstimator(num_symbols=3, laplace=True)
        est.add_value(0)
        est.add_value(1)
        est.add_value(2)

        # With Laplace: (1 + 1) / (3 + 3) = 2/6 = 1/3
        for i in range(3):
            np.testing.assert_allclose(est.get_probability(i), 1.0 / 3.0, atol=1e-10)

    def test_uniform_distribution_without_laplace(self):
        """Test uniform distribution after equal additions without Laplace smoothing."""
        est = DiscreteEstimator(num_symbols=3, laplace=False)
        est.add_value(0)
        est.add_value(1)
        est.add_value(2)

        # Without Laplace: 1/3
        for i in range(3):
            np.testing.assert_allclose(est.get_probability(i), 1.0 / 3.0, atol=1e-10)

    def test_laplace_smoothing_nonzero_for_unseen(self):
        """Test that Laplace smoothing gives nonzero probability for unseen symbols."""
        est = DiscreteEstimator(num_symbols=3, laplace=True)
        est.add_value(0)

        # Unseen symbol with Laplace: (0 + 1) / (1 + 3) = 1/4
        prob_unseen = est.get_probability(2)
        assert prob_unseen > 0.0
        np.testing.assert_allclose(prob_unseen, 0.25, atol=1e-10)

    def test_no_laplace_zero_for_unseen(self):
        """Test that without Laplace smoothing, unseen symbols have zero probability."""
        est = DiscreteEstimator(num_symbols=3, laplace=False)
        est.add_value(0)

        assert est.get_probability(2) == 0.0

    def test_probability_out_of_bounds_is_zero(self):
        """Test that querying out-of-bounds symbol returns zero."""
        est = DiscreteEstimator(num_symbols=3)
        est.add_value(0)

        assert est.get_probability(-1) == 0.0
        assert est.get_probability(3) == 0.0
        assert est.get_probability(100) == 0.0

    def test_probability_sums_to_one_laplace(self):
        """Test that probabilities sum to approximately 1 with Laplace smoothing."""
        est = DiscreteEstimator(num_symbols=4, laplace=True)
        est.add_value(0)
        est.add_value(0)
        est.add_value(1)
        est.add_value(3)

        total_prob = sum(est.get_probability(i) for i in range(4))
        np.testing.assert_allclose(total_prob, 1.0, atol=1e-10)

    def test_probability_sums_to_one_no_laplace(self):
        """Test that probabilities sum to approximately 1 without Laplace smoothing."""
        est = DiscreteEstimator(num_symbols=4, laplace=False)
        est.add_value(0)
        est.add_value(0)
        est.add_value(1)
        est.add_value(3)

        total_prob = sum(est.get_probability(i) for i in range(4))
        np.testing.assert_allclose(total_prob, 1.0, atol=1e-10)


class TestDiscreteEstimatorEdgeCases:
    """Edge case tests."""

    def test_zero_total_count_no_laplace(self):
        """Test probability with zero total count and no Laplace smoothing."""
        est = DiscreteEstimator(num_symbols=3, laplace=False)

        assert est.get_probability(0) == 0.0

    def test_zero_total_count_with_laplace(self):
        """Test probability with zero total count and Laplace smoothing."""
        est = DiscreteEstimator(num_symbols=3, laplace=True)

        # With Laplace: (0 + 1) / (0 + 3) = 1/3
        np.testing.assert_allclose(est.get_probability(0), 1.0 / 3.0, atol=1e-10)

    def test_float_value_truncated(self):
        """Test that float values are truncated to integer indices."""
        est = DiscreteEstimator(num_symbols=3)
        est.add_value(1.7)

        assert est.get_count(1) == 1.0
        assert est.get_count(2) == 0.0

    def test_get_count_out_of_bounds(self):
        """Test get_count returns 0 for out-of-bounds symbol."""
        est = DiscreteEstimator(num_symbols=3)

        assert est.get_count(-1) == 0.0
        assert est.get_count(5) == 0.0
