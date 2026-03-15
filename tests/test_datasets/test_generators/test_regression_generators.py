"""Tests for regression data generators."""

import numpy as np
import pytest

from tuiml.datasets.generators.regression.friedman import Friedman
from tuiml.datasets.generators.regression.mexican_hat import MexicanHat
from tuiml.datasets.generators.regression.sine import Sine
from tuiml.base.generators import GeneratedData


# ===================================================================
# Friedman generator tests
# ===================================================================

class TestFriedman:
    """Tests for the Friedman regression generator."""

    def test_default_output_shape(self):
        """Default Friedman should produce (100, 10) X and (100,) y."""
        gen = Friedman(random_state=42)
        data = gen.generate()
        assert isinstance(data, GeneratedData)
        assert data.X.shape == (100, 10)
        assert data.y.shape == (100,)

    def test_custom_n_samples(self):
        """n_samples should control number of output rows."""
        gen = Friedman(n_samples=500, random_state=42)
        data = gen.generate()
        assert data.n_samples == 500

    def test_custom_n_features(self):
        """n_features should control columns (minimum 5 for function 1)."""
        gen = Friedman(n_features=15, random_state=42)
        data = gen.generate()
        assert data.n_features == 15

    def test_minimum_features_function_1(self):
        """Function 1 requires at least 5 features; smaller values are clamped."""
        gen = Friedman(n_features=2, function=1, random_state=42)
        data = gen.generate()
        assert data.n_features >= 5

    def test_minimum_features_function_2(self):
        """Function 2 requires at least 4 features."""
        gen = Friedman(n_features=2, function=2, random_state=42)
        data = gen.generate()
        assert data.n_features >= 4

    def test_continuous_target(self):
        """Target values should be continuous floats, not integers."""
        gen = Friedman(n_samples=200, random_state=42)
        data = gen.generate()
        # Check that y is not all integers
        assert data.y.dtype in [np.float64, np.float32]
        # There should be many distinct values
        assert len(set(data.y)) > 10

    def test_function_2(self):
        """Function 2 should produce data without errors."""
        gen = Friedman(n_samples=50, function=2, random_state=42)
        data = gen.generate()
        assert data.n_samples == 50
        assert not np.any(np.isnan(data.y))

    def test_function_3(self):
        """Function 3 should produce data without errors."""
        gen = Friedman(n_samples=50, function=3, random_state=42)
        data = gen.generate()
        assert data.n_samples == 50
        assert not np.any(np.isnan(data.y))

    def test_noise_increases_variance(self):
        """Adding noise should increase variance of y."""
        gen_clean = Friedman(n_samples=500, noise=0.0, random_state=42)
        gen_noisy = Friedman(n_samples=500, noise=5.0, random_state=42)
        y_clean = gen_clean.generate().y
        y_noisy = gen_noisy.generate().y
        assert np.var(y_noisy) > np.var(y_clean)

    def test_random_state_reproducibility(self):
        """Same random_state should produce identical data."""
        gen1 = Friedman(n_samples=50, random_state=123)
        gen2 = Friedman(n_samples=50, random_state=123)
        data1 = gen1.generate()
        data2 = gen2.generate()
        np.testing.assert_array_equal(data1.X, data2.X)
        np.testing.assert_array_equal(data1.y, data2.y)

    def test_feature_names(self):
        """Feature names should be x0, x1, ..., xN."""
        gen = Friedman(n_features=5, random_state=42)
        data = gen.generate()
        assert data.feature_names == ["x0", "x1", "x2", "x3", "x4"]


# ===================================================================
# MexicanHat generator tests
# ===================================================================

class TestMexicanHat:
    """Tests for the MexicanHat regression generator."""

    def test_default_output_shape(self):
        """Default MexicanHat should produce (100, 2) X and (100,) y."""
        gen = MexicanHat(random_state=42)
        data = gen.generate()
        assert data.X.shape == (100, 2)
        assert data.y.shape == (100,)

    def test_custom_n_samples(self):
        """n_samples should control output size."""
        gen = MexicanHat(n_samples=300, random_state=42)
        data = gen.generate()
        assert data.n_samples == 300

    def test_custom_n_features(self):
        """n_features should control dimensionality."""
        gen = MexicanHat(n_features=5, random_state=42)
        data = gen.generate()
        assert data.n_features == 5

    def test_continuous_target(self):
        """Target values should be continuous floats."""
        gen = MexicanHat(n_samples=200, random_state=42)
        data = gen.generate()
        assert data.y.dtype in [np.float64, np.float32]
        assert len(set(data.y)) > 10

    def test_amplitude_scales_output(self):
        """Larger amplitude should produce larger range of y values."""
        gen_small = MexicanHat(n_samples=500, amplitude=1.0, random_state=42)
        gen_large = MexicanHat(n_samples=500, amplitude=10.0, random_state=42)
        y_small = gen_small.generate().y
        y_large = gen_large.generate().y
        assert np.max(np.abs(y_large)) > np.max(np.abs(y_small))

    def test_sigma_parameter(self):
        """Different sigma should produce different data distributions."""
        gen1 = MexicanHat(n_samples=100, sigma=0.5, random_state=42)
        gen2 = MexicanHat(n_samples=100, sigma=2.0, random_state=42)
        data1 = gen1.generate()
        data2 = gen2.generate()
        # Different sigma means X ranges differ (sigma * 4 = range)
        assert not np.array_equal(data1.X, data2.X)

    def test_noise_increases_variance(self):
        """Adding noise should increase the variance of y."""
        gen_clean = MexicanHat(n_samples=500, noise=0.0, random_state=42)
        gen_noisy = MexicanHat(n_samples=500, noise=1.0, random_state=42)
        y_clean = gen_clean.generate().y
        y_noisy = gen_noisy.generate().y
        assert np.var(y_noisy) > np.var(y_clean)

    def test_random_state_reproducibility(self):
        """Same random_state should produce identical data."""
        gen1 = MexicanHat(n_samples=50, random_state=99)
        gen2 = MexicanHat(n_samples=50, random_state=99)
        data1 = gen1.generate()
        data2 = gen2.generate()
        np.testing.assert_array_equal(data1.X, data2.X)
        np.testing.assert_array_equal(data1.y, data2.y)

    def test_feature_names(self):
        """Feature names should be x0, x1."""
        gen = MexicanHat(random_state=42)
        data = gen.generate()
        assert data.feature_names == ["x0", "x1"]


# ===================================================================
# Sine generator tests
# ===================================================================

class TestSine:
    """Tests for the Sine regression generator."""

    def test_default_output_shape(self):
        """Default Sine should produce (100, 1) X and (100,) y."""
        gen = Sine(random_state=42)
        data = gen.generate()
        assert data.X.shape == (100, 1)
        assert data.y.shape == (100,)

    def test_custom_n_samples(self):
        """n_samples should control number of output rows."""
        gen = Sine(n_samples=250, random_state=42)
        data = gen.generate()
        assert data.n_samples == 250

    def test_custom_n_features(self):
        """n_features > 1 should produce multi-dimensional input."""
        gen = Sine(n_features=3, random_state=42)
        data = gen.generate()
        assert data.n_features == 3

    def test_continuous_target(self):
        """Target should be continuous sine values."""
        gen = Sine(n_samples=200, random_state=42)
        data = gen.generate()
        assert data.y.dtype in [np.float64, np.float32]
        assert len(set(data.y)) > 10

    def test_amplitude_scales_output(self):
        """Larger amplitude should produce larger y range."""
        gen_small = Sine(n_samples=200, amplitude=1.0, noise=0.0, random_state=42)
        gen_large = Sine(n_samples=200, amplitude=5.0, noise=0.0, random_state=42)
        y_small = gen_small.generate().y
        y_large = gen_large.generate().y
        assert np.max(np.abs(y_large)) > np.max(np.abs(y_small))

    def test_offset_shifts_output(self):
        """Offset should shift all y values."""
        gen_no_offset = Sine(n_samples=100, offset=0.0, noise=0.0, random_state=42)
        gen_offset = Sine(n_samples=100, offset=10.0, noise=0.0, random_state=42)
        y_base = gen_no_offset.generate().y
        y_shifted = gen_offset.generate().y
        np.testing.assert_array_almost_equal(y_shifted, y_base + 10.0)

    def test_noise_adds_variation(self):
        """Adding noise should increase variance of y."""
        gen_clean = Sine(n_samples=500, noise=0.0, random_state=42)
        gen_noisy = Sine(n_samples=500, noise=1.0, random_state=42)
        y_clean = gen_clean.generate().y
        y_noisy = gen_noisy.generate().y
        assert np.var(y_noisy) > np.var(y_clean)

    def test_random_state_reproducibility(self):
        """Same random_state should produce identical data."""
        gen1 = Sine(n_samples=50, random_state=77)
        gen2 = Sine(n_samples=50, random_state=77)
        data1 = gen1.generate()
        data2 = gen2.generate()
        np.testing.assert_array_equal(data1.X, data2.X)
        np.testing.assert_array_equal(data1.y, data2.y)

    def test_feature_names(self):
        """Feature names should be x0 for single feature."""
        gen = Sine(random_state=42)
        data = gen.generate()
        assert data.feature_names == ["x0"]

    def test_y_bounded_without_noise(self):
        """Without noise, y should be bounded by [-amplitude, amplitude] + offset."""
        gen = Sine(n_samples=1000, amplitude=2.0, offset=0.0, noise=0.0, random_state=42)
        data = gen.generate()
        assert np.all(data.y >= -2.0 - 1e-10)
        assert np.all(data.y <= 2.0 + 1e-10)
