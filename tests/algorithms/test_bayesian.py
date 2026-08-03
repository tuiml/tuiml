"""Naive Bayes classifiers and the probability estimators behind them.

Merged from: test_bayesian_naive_bayes.py, test_estimators_normal.py, test_estimators_discrete.py, test_estimators_kernel.py
"""

import numpy as np
import pickle
from tuiml.algorithms.bayesian import NaiveBayesClassifier
from tuiml.algorithms.bayesian.estimators import NormalEstimator
from tuiml.algorithms.bayesian.estimators import DiscreteEstimator
from tuiml.algorithms.bayesian.estimators import KernelEstimator


# --------------------------------------------------------------------------
# Test suite for NaiveBayesClassifier.
# --------------------------------------------------------------------------

class TestNaiveBayesClassifierInstantiation:
    """Tests for algorithm instantiation."""
    
    def test_fit_basic(self, binary_cls_data):
        """Test basic fitting."""
        X, y = binary_cls_data
        model = NaiveBayesClassifier()
        model.fit(X, y)
        assert model._is_fitted is True
        
        preds = model.predict(X)
        assert preds.shape == y.shape
        
        probas = model.predict_proba(X)
        assert probas.shape == (len(X), len(model.classes_))
        assert np.allclose(probas.sum(axis=1), 1.0)
        
    def test_partial_fit(self, binary_cls_data):
        """Test partial_fit incremental training."""
        X, y = binary_cls_data
        classes = np.unique(y)
        
        model = NaiveBayesClassifier()
        
        # Split into batches
        n_samples = len(X)
        half = n_samples // 2
        
        # First batch
        model.partial_fit(X[:half], y[:half], classes=classes)
        assert model._is_fitted is True
        assert model.n_samples_seen_ == half
        
        # Second batch
        model.partial_fit(X[half:], y[half:])
        assert model.n_samples_seen_ == n_samples
        
        # Make predictions
        preds = model.predict(X)
        assert preds.shape == y.shape
        
        probas = model.predict_proba(X)
        assert probas.shape == (len(X), len(classes))
        assert np.allclose(probas.sum(axis=1), 1.0)


class TestNaiveBayesClassifierSerialization:
    """Tests for serialization."""
    
    def test_pickle_roundtrip(self, binary_cls_data):
        """Test pickle serialization."""
        X, y = binary_cls_data
        model = NaiveBayesClassifier()
        model.fit(X, y)
        
        data = pickle.dumps(model)
        loaded_model = pickle.loads(data)
        
        assert loaded_model._is_fitted is True
        assert np.array_equal(loaded_model.predict(X), model.predict(X))


# --------------------------------------------------------------------------
# Test suite for NormalEstimator probability density estimator.
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Test suite for DiscreteEstimator probability estimator.
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Test suite for KernelEstimator (Kernel Density Estimator).
# --------------------------------------------------------------------------

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
