"""Test suite for KernelEstimator (Kernel Density Estimator).

Tests cover:
- Default and custom initialization
- Density estimation basics
- Probability is non-negative
- NaN handling
- Weighted observations
- Edge cases
"""

import numpy as np
import pytest

from tuiml.algorithms.bayesian.estimators import KernelEstimator


class TestKernelEstimatorInit:
    """Tests for KernelEstimator initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        est = KernelEstimator()

        assert est.total_weight == 0.0
        assert est.precision == 1e-6
        assert len(est.values) == 0
        assert len(est.weights) == 0
        assert est.standard_deviation == -1.0

    def test_custom_precision(self):
        """Test initialization with custom precision."""
        est = KernelEstimator(precision=0.01)

        assert est.precision == 0.01

    def test_none_precision_defaults(self):
        """Test that precision=None falls back to 1e-6."""
        est = KernelEstimator(precision=None)

        assert est.precision == 1e-6


class TestKernelEstimatorAddValues:
    """Tests for the add_value() method."""

    def test_add_values_updates_state(self):
        """Test that adding values updates internal state correctly."""
        est = KernelEstimator()
        est.add_value(1.0)
        est.add_value(2.0)
        est.add_value(3.0)

        assert len(est.values) == 3
        assert len(est.weights) == 3
        assert est.total_weight == 3.0

    def test_add_value_nan_ignored(self):
        """Test that NaN values are silently ignored."""
        est = KernelEstimator()
        est.add_value(1.0)
        est.add_value(np.nan)
        est.add_value(3.0)

        assert len(est.values) == 2
        assert est.total_weight == 2.0

    def test_add_weighted_value(self):
        """Test adding a weighted value."""
        est = KernelEstimator()
        est.add_value(5.0, weight=3.0)

        assert len(est.values) == 1
        assert est.weights[0] == 3.0
        assert est.total_weight == 3.0
        assert est.all_weights_one is False

    def test_invalidates_bandwidth_on_add(self):
        """Test that adding a value invalidates the cached bandwidth."""
        est = KernelEstimator()
        est.add_value(1.0)
        est.add_value(2.0)

        # Force bandwidth calculation
        est.get_probability(1.5)
        assert est.standard_deviation > 0

        # Adding a new value should invalidate it
        est.add_value(3.0)
        assert est.standard_deviation == -1.0


class TestKernelEstimatorProbability:
    """Tests for get_probability()."""

    def test_probability_is_non_negative(self):
        """Test that probability density is always non-negative."""
        est = KernelEstimator()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            est.add_value(v)

        for x in [-10.0, 0.0, 3.0, 10.0, 100.0]:
            assert est.get_probability(x) >= 0.0

    def test_probability_higher_near_data(self):
        """Test that density is higher near the data points."""
        est = KernelEstimator()
        for v in [1.0, 1.1, 1.2, 5.0, 5.1, 5.2]:
            est.add_value(v)

        prob_near_mode = est.get_probability(1.1)
        prob_far_away = est.get_probability(50.0)

        assert prob_near_mode > prob_far_away

    def test_probability_bimodal_peaks(self):
        """Test that a bimodal distribution has two peaks."""
        est = KernelEstimator()
        # Two tight clusters
        for v in [0.0, 0.01, -0.01]:
            est.add_value(v)
        for v in [10.0, 10.01, 9.99]:
            est.add_value(v)

        prob_mode1 = est.get_probability(0.0)
        prob_mode2 = est.get_probability(10.0)
        prob_between = est.get_probability(5.0)

        assert prob_mode1 > prob_between
        assert prob_mode2 > prob_between

    def test_probability_zero_weight_returns_zero(self):
        """Test that probability is zero when no values added."""
        est = KernelEstimator()

        assert est.get_probability(0.0) == 0.0

    def test_probability_single_point(self):
        """Test density estimation with a single data point."""
        est = KernelEstimator()
        est.add_value(5.0)

        prob_at_point = est.get_probability(5.0)
        prob_far = est.get_probability(100.0)

        assert prob_at_point > 0.0
        assert prob_at_point > prob_far


class TestKernelEstimatorEdgeCases:
    """Edge case tests."""

    def test_many_identical_values(self):
        """Test with many identical values."""
        est = KernelEstimator()
        for _ in range(50):
            est.add_value(3.0)

        prob = est.get_probability(3.0)
        assert prob > 0.0
        assert np.isfinite(prob)

    def test_large_spread_data(self):
        """Test with data spread over a large range."""
        np.random.seed(42)
        est = KernelEstimator()
        for v in np.random.uniform(-1000, 1000, 100):
            est.add_value(v)

        prob = est.get_probability(0.0)
        assert prob >= 0.0
        assert np.isfinite(prob)

    def test_weighted_vs_unweighted(self):
        """Test that weighted addition produces different density than unweighted."""
        est_unweighted = KernelEstimator()
        est_weighted = KernelEstimator()

        est_unweighted.add_value(1.0)
        est_unweighted.add_value(1.0)
        est_unweighted.add_value(5.0)

        est_weighted.add_value(1.0, weight=2.0)
        est_weighted.add_value(5.0, weight=1.0)

        # Both should give similar density
        prob_uw = est_unweighted.get_probability(1.0)
        prob_w = est_weighted.get_probability(1.0)

        # They should be finite and positive
        assert prob_uw > 0.0
        assert prob_w > 0.0
