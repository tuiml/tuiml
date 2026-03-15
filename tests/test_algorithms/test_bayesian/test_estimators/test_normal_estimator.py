"""Test suite for NormalEstimator probability density estimator.

Tests cover:
- Default and custom initialization
- Adding values and verifying mean/std
- Probability density at the mean is highest
- NaN handling
- Weighted values
- Zero count edge case
"""

import numpy as np
import pytest

from tuiml.algorithms.bayesian.estimators import NormalEstimator


class TestNormalEstimatorInit:
    """Tests for NormalEstimator initialization."""

    def test_default_initialization(self):
        """Test default initialization sets correct defaults."""
        est = NormalEstimator()

        assert est.sum == 0.0
        assert est.sum_sq == 0.0
        assert est.count == 0.0
        assert est.precision == 1e-6

    def test_custom_precision(self):
        """Test initialization with a custom precision value."""
        est = NormalEstimator(precision=0.01)

        assert est.precision == 0.01

    def test_none_precision_defaults(self):
        """Test that precision=None falls back to 1e-6."""
        est = NormalEstimator(precision=None)

        assert est.precision == 1e-6


class TestNormalEstimatorAddValues:
    """Tests for the add_value() method and resulting statistics."""

    def test_add_values_mean(self):
        """Test that mean is correctly computed after adding values."""
        est = NormalEstimator()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            est.add_value(v)

        np.testing.assert_allclose(est.get_mean(), 3.0, atol=1e-10)

    def test_add_values_std_dev(self):
        """Test that standard deviation is correctly computed."""
        est = NormalEstimator()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            est.add_value(v)

        # Population std of [1,2,3,4,5] = sqrt(2.0) = 1.4142...
        expected_std = np.sqrt(2.0)
        np.testing.assert_allclose(est.get_std_dev(), expected_std, atol=1e-10)

    def test_add_single_value(self):
        """Test adding a single value results in mean equal to that value."""
        est = NormalEstimator()
        est.add_value(42.0)

        np.testing.assert_allclose(est.get_mean(), 42.0, atol=1e-10)

    def test_add_identical_values(self):
        """Test adding identical values gives correct mean and minimal std."""
        est = NormalEstimator()
        for _ in range(10):
            est.add_value(5.0)

        np.testing.assert_allclose(est.get_mean(), 5.0, atol=1e-10)
        # Variance should be floored at precision
        assert est.get_std_dev() >= 0


class TestNormalEstimatorProbability:
    """Tests for get_probability()."""

    def test_probability_at_mean_is_highest(self):
        """Test that probability density at the mean exceeds density at other points."""
        est = NormalEstimator()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            est.add_value(v)

        prob_at_mean = est.get_probability(3.0)
        prob_at_offset = est.get_probability(5.0)

        assert prob_at_mean > prob_at_offset

    def test_probability_is_non_negative(self):
        """Test that probability density is always non-negative."""
        est = NormalEstimator()
        for v in [1.0, 2.0, 3.0]:
            est.add_value(v)

        for x in [-10.0, 0.0, 2.0, 5.0, 100.0]:
            assert est.get_probability(x) >= 0.0

    def test_probability_symmetric_around_mean(self):
        """Test that the density is symmetric around the mean."""
        est = NormalEstimator()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            est.add_value(v)

        mean = est.get_mean()
        offset = 1.0
        prob_left = est.get_probability(mean - offset)
        prob_right = est.get_probability(mean + offset)

        np.testing.assert_allclose(prob_left, prob_right, atol=1e-10)

    def test_probability_far_from_mean_is_small(self):
        """Test that probability far from the mean is very small."""
        est = NormalEstimator()
        for v in [0.0, 0.1, -0.1, 0.05, -0.05]:
            est.add_value(v)

        prob_far = est.get_probability(100.0)
        assert prob_far < 1e-6


class TestNormalEstimatorNaN:
    """Tests for NaN handling."""

    def test_add_nan_is_ignored(self):
        """Test that adding NaN does not affect the estimator state."""
        est = NormalEstimator()
        est.add_value(1.0)
        est.add_value(np.nan)
        est.add_value(3.0)

        assert est.count == 2.0
        np.testing.assert_allclose(est.get_mean(), 2.0, atol=1e-10)

    def test_probability_of_nan_is_zero(self):
        """Test that querying probability for NaN returns 0."""
        est = NormalEstimator()
        est.add_value(1.0)
        est.add_value(2.0)

        assert est.get_probability(np.nan) == 0.0


class TestNormalEstimatorWeighted:
    """Tests for weighted value addition."""

    def test_weighted_mean(self):
        """Test that weighted values produce the correct weighted mean."""
        est = NormalEstimator()
        est.add_value(1.0, weight=2.0)
        est.add_value(3.0, weight=2.0)

        # Weighted mean: (1*2 + 3*2) / (2+2) = 8/4 = 2.0
        np.testing.assert_allclose(est.get_mean(), 2.0, atol=1e-10)

    def test_weighted_count(self):
        """Test that weighted values accumulate the correct total count."""
        est = NormalEstimator()
        est.add_value(1.0, weight=3.0)
        est.add_value(2.0, weight=7.0)

        assert est.count == 10.0


class TestNormalEstimatorEdgeCases:
    """Edge case tests."""

    def test_zero_count_probability(self):
        """Test probability when no values have been added."""
        est = NormalEstimator()

        # Should not raise and should return a finite value
        prob = est.get_probability(0.0)
        assert np.isfinite(prob)

    def test_zero_count_mean(self):
        """Test mean when no values have been added."""
        est = NormalEstimator()

        assert est.get_mean() == 0.0

    def test_precision_floor(self):
        """Test that variance is floored at precision for identical values."""
        est = NormalEstimator(precision=0.01)
        for _ in range(100):
            est.add_value(5.0)

        # Std dev should be at least sqrt(precision)
        assert est.get_std_dev() >= np.sqrt(0.01) - 1e-10
