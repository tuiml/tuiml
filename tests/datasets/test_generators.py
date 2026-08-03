"""Synthetic data generators.

Merged from: test_generators_classification.py, test_generators_regression.py, test_generators_clustering.py
"""

import numpy as np
from tuiml.datasets.generators.classification.hyperplane import Hyperplane
from tuiml.datasets.generators.classification.agrawal import Agrawal
from tuiml.datasets.generators.classification.led import LED
from tuiml.datasets.generators.classification.random_rbf import RandomRBF
from tuiml.base.generators import GeneratedData
from tuiml.datasets.generators.regression.friedman import Friedman
from tuiml.datasets.generators.regression.mexican_hat import MexicanHat
from tuiml.datasets.generators.regression.sine import Sine
from tuiml.datasets.generators.clustering.blobs import Blobs
from tuiml.datasets.generators.clustering.moons import Moons
from tuiml.datasets.generators.clustering.circles import Circles
from tuiml.datasets.generators.clustering.swiss_roll import SwissRoll


# --------------------------------------------------------------------------
# Tests for classification data generators.
# --------------------------------------------------------------------------

class TestHyperplane:
    """Tests for the Hyperplane classification generator."""

    def test_default_output_shape(self):
        """Default generator should produce (100, 10) X and (100,) y."""
        gen = Hyperplane(random_state=42)
        data = gen.generate()
        assert isinstance(data, GeneratedData)
        assert data.X.shape == (100, 10)
        assert data.y.shape == (100,)

    def test_custom_n_samples(self):
        """n_samples parameter should control number of rows."""
        gen = Hyperplane(n_samples=500, random_state=42)
        data = gen.generate()
        assert data.X.shape[0] == 500
        assert data.y.shape[0] == 500

    def test_custom_n_features(self):
        """n_features parameter should control number of columns."""
        gen = Hyperplane(n_features=5, random_state=42)
        data = gen.generate()
        assert data.X.shape[1] == 5

    def test_binary_labels(self):
        """Labels should be binary (0 or 1)."""
        gen = Hyperplane(n_samples=200, random_state=42)
        data = gen.generate()
        unique = set(data.y)
        assert unique.issubset({0, 1})

    def test_random_state_reproducibility(self):
        """Same random_state should produce identical results."""
        gen1 = Hyperplane(n_samples=50, random_state=123)
        gen2 = Hyperplane(n_samples=50, random_state=123)
        data1 = gen1.generate()
        data2 = gen2.generate()
        np.testing.assert_array_equal(data1.X, data2.X)
        np.testing.assert_array_equal(data1.y, data2.y)

    def test_different_random_state_produces_different_data(self):
        """Different random states should produce different data."""
        gen1 = Hyperplane(n_samples=50, random_state=1)
        gen2 = Hyperplane(n_samples=50, random_state=2)
        data1 = gen1.generate()
        data2 = gen2.generate()
        assert not np.array_equal(data1.X, data2.X)

    def test_feature_names(self):
        """Feature names should be x0, x1, ..., xN."""
        gen = Hyperplane(n_features=3, random_state=42)
        data = gen.generate()
        assert data.feature_names == ["x0", "x1", "x2"]

    def test_target_names(self):
        """Target names should be class0, class1."""
        gen = Hyperplane(random_state=42)
        data = gen.generate()
        assert data.target_names == ["class0", "class1"]

    def test_noise_flips_labels(self):
        """With noise=1.0 all labels should be flipped."""
        gen_no_noise = Hyperplane(n_samples=100, noise=0.0, random_state=42)
        gen_full_noise = Hyperplane(n_samples=100, noise=1.0, random_state=42)
        data_clean = gen_no_noise.generate()
        data_noisy = gen_full_noise.generate()
        # With noise=1.0, every label should be flipped
        np.testing.assert_array_equal(data_noisy.y, 1 - data_clean.y)

    def test_callable_interface(self):
        """Generator should work as a callable returning GeneratedData."""
        gen = Hyperplane(n_samples=10, random_state=42)
        data = gen()
        assert isinstance(data, GeneratedData)

    def test_callable_return_X_y(self):
        """Generator callable with return_X_y=True should return tuple."""
        gen = Hyperplane(n_samples=10, random_state=42)
        X, y = gen(return_X_y=True)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]


class TestAgrawal:
    """Tests for the Agrawal classification generator."""

    def test_default_output_shape(self):
        """Default Agrawal should produce (100, 9) X."""
        gen = Agrawal(random_state=42)
        data = gen.generate()
        assert data.X.shape == (100, 9)
        assert data.y.shape == (100,)

    def test_custom_n_samples(self):
        """n_samples should control output size."""
        gen = Agrawal(n_samples=300, random_state=42)
        data = gen.generate()
        assert data.n_samples == 300

    def test_fixed_9_features(self):
        """Agrawal always has 9 features regardless of parameters."""
        gen = Agrawal(n_samples=50, random_state=42)
        data = gen.generate()
        assert data.n_features == 9

    def test_feature_names_match(self):
        """Feature names should match FEATURE_NAMES constant."""
        gen = Agrawal(random_state=42)
        data = gen.generate()
        expected = ['salary', 'commission', 'age', 'education_level', 'car',
                    'zipcode', 'house_value', 'years_house', 'loan']
        assert data.feature_names == expected

    def test_binary_labels(self):
        """Agrawal produces binary classification (0 or 1)."""
        gen = Agrawal(n_samples=200, random_state=42)
        data = gen.generate()
        assert set(data.y).issubset({0, 1})

    def test_different_functions(self):
        """Different function indices should produce different class distributions."""
        data1 = Agrawal(n_samples=200, function=1, random_state=42).generate()
        data5 = Agrawal(n_samples=200, function=5, random_state=42).generate()
        # The features are generated with same seed but classification differs
        assert not np.array_equal(data1.y, data5.y)

    def test_function_clamped(self):
        """Function values outside 1-10 should be clamped."""
        gen = Agrawal(function=0, random_state=42)
        assert gen.function == 1
        gen = Agrawal(function=15, random_state=42)
        assert gen.function == 10

    def test_random_state_reproducibility(self):
        """Same random_state should produce identical data."""
        gen1 = Agrawal(n_samples=50, function=3, random_state=99)
        gen2 = Agrawal(n_samples=50, function=3, random_state=99)
        data1 = gen1.generate()
        data2 = gen2.generate()
        np.testing.assert_array_equal(data1.X, data2.X)
        np.testing.assert_array_equal(data1.y, data2.y)

    def test_target_names(self):
        """Target names should be group_A and group_B."""
        gen = Agrawal(random_state=42)
        data = gen.generate()
        assert data.target_names == ['group_A', 'group_B']


class TestLED:
    """Tests for the LED (Light Emitting Diode) classification generator."""

    def test_default_output_shape(self):
        """Default LED should produce (100, 24) X (7 segments + 17 irrelevant)."""
        gen = LED(random_state=42)
        data = gen.generate()
        assert data.X.shape == (100, 24)
        assert data.y.shape == (100,)

    def test_custom_n_samples(self):
        """n_samples should control output row count."""
        gen = LED(n_samples=500, random_state=42)
        data = gen.generate()
        assert data.n_samples == 500

    def test_no_irrelevant_features(self):
        """With n_irrelevant=0, only 7 segment features should exist."""
        gen = LED(n_irrelevant=0, random_state=42)
        data = gen.generate()
        assert data.n_features == 7

    def test_labels_are_digits(self):
        """Labels should be integers in 0-9."""
        gen = LED(n_samples=500, random_state=42)
        data = gen.generate()
        assert set(data.y).issubset(set(range(10)))

    def test_ten_target_names(self):
        """Target names should be 0 through 9 as strings."""
        gen = LED(random_state=42)
        data = gen.generate()
        assert data.target_names == [str(i) for i in range(10)]

    def test_random_state_reproducibility(self):
        """Same random_state should produce identical data."""
        gen1 = LED(n_samples=50, random_state=77)
        gen2 = LED(n_samples=50, random_state=77)
        data1 = gen1.generate()
        data2 = gen2.generate()
        np.testing.assert_array_equal(data1.X, data2.X)
        np.testing.assert_array_equal(data1.y, data2.y)

    def test_feature_names_segments_and_irrelevant(self):
        """Feature names should include seg0-seg6 and irr0-irr16."""
        gen = LED(random_state=42)
        data = gen.generate()
        assert data.feature_names[0] == "seg0"
        assert data.feature_names[6] == "seg6"
        assert data.feature_names[7] == "irr0"
        assert data.feature_names[-1] == "irr16"


class TestRandomRBF:
    """Tests for the RandomRBF classification generator."""

    def test_default_output_shape(self):
        """Default RandomRBF should produce (100, 10) X."""
        gen = RandomRBF(random_state=42)
        data = gen.generate()
        assert data.X.shape == (100, 10)
        assert data.y.shape == (100,)

    def test_custom_parameters(self):
        """Custom n_samples, n_features, n_classes should affect output."""
        gen = RandomRBF(n_samples=200, n_features=5, n_classes=4, random_state=42)
        data = gen.generate()
        assert data.n_samples == 200
        assert data.n_features == 5
        unique_classes = set(data.y)
        # Not all classes may appear with small n_samples, but we can check bounds
        assert max(unique_classes) < 4

    def test_binary_default(self):
        """Default n_classes=2 should produce binary classification."""
        gen = RandomRBF(n_samples=200, random_state=42)
        data = gen.generate()
        assert set(data.y).issubset({0, 1})

    def test_random_state_reproducibility(self):
        """Same random_state should produce identical data."""
        gen1 = RandomRBF(n_samples=50, random_state=55)
        gen2 = RandomRBF(n_samples=50, random_state=55)
        data1 = gen1.generate()
        data2 = gen2.generate()
        np.testing.assert_array_equal(data1.X, data2.X)
        np.testing.assert_array_equal(data1.y, data2.y)

    def test_feature_names(self):
        """Feature names should be attr0, attr1, etc."""
        gen = RandomRBF(n_features=3, random_state=42)
        data = gen.generate()
        assert data.feature_names == ["attr0", "attr1", "attr2"]

    def test_target_names(self):
        """Target names should be class0, class1, etc."""
        gen = RandomRBF(n_classes=3, random_state=42)
        data = gen.generate()
        assert data.target_names == ["class0", "class1", "class2"]

    def test_n_centroids_parameter(self):
        """n_centroids should be stored without error."""
        gen = RandomRBF(n_centroids=10, random_state=42)
        assert gen.n_centroids == 10
        data = gen.generate()
        assert data.n_samples == 100  # default


# --------------------------------------------------------------------------
# Tests for regression data generators.
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Tests for clustering data generators.
# --------------------------------------------------------------------------

class TestBlobs:
    """Tests for the Gaussian Blobs clustering generator."""

    def test_default_output_shape(self):
        """Default Blobs should produce (100, 2) X and (100,) y."""
        gen = Blobs(random_state=42)
        data = gen.generate()
        assert isinstance(data, GeneratedData)
        assert data.X.shape == (100, 2)
        assert data.y.shape == (100,)

    def test_custom_n_samples(self):
        """n_samples should control number of output rows."""
        gen = Blobs(n_samples=500, random_state=42)
        data = gen.generate()
        assert data.n_samples == 500

    def test_custom_n_features(self):
        """n_features should control dimensionality."""
        gen = Blobs(n_features=5, random_state=42)
        data = gen.generate()
        assert data.n_features == 5

    def test_custom_n_clusters(self):
        """n_clusters should control number of distinct cluster labels."""
        gen = Blobs(n_samples=300, n_clusters=5, random_state=42)
        data = gen.generate()
        unique = set(data.y)
        assert len(unique) == 5

    def test_labels_are_cluster_indices(self):
        """Labels should be integers 0 to n_clusters-1."""
        gen = Blobs(n_clusters=4, n_samples=200, random_state=42)
        data = gen.generate()
        assert set(data.y).issubset(set(range(4)))

    def test_custom_centers(self):
        """Providing explicit centers should be used."""
        centers = np.array([[0, 0], [10, 10], [20, 20]])
        gen = Blobs(n_samples=300, centers=centers, random_state=42)
        data = gen.generate()
        assert data.n_samples == 300
        assert len(set(data.y)) == 3

    def test_cluster_std_scalar(self):
        """Scalar cluster_std should apply to all clusters."""
        gen = Blobs(n_samples=200, n_clusters=3, cluster_std=0.1, random_state=42)
        data = gen.generate()
        # With small std, clusters should be tight
        assert data.n_samples == 200

    def test_cluster_std_list(self):
        """List cluster_std should apply per-cluster."""
        gen = Blobs(n_samples=300, n_clusters=3, cluster_std=[0.1, 0.5, 1.0],
                    random_state=42)
        data = gen.generate()
        assert data.n_samples == 300

    def test_random_state_reproducibility(self):
        """Same random_state should produce identical data."""
        gen1 = Blobs(n_samples=50, random_state=88)
        gen2 = Blobs(n_samples=50, random_state=88)
        data1 = gen1.generate()
        data2 = gen2.generate()
        np.testing.assert_array_equal(data1.X, data2.X)
        np.testing.assert_array_equal(data1.y, data2.y)

    def test_feature_names(self):
        """Feature names should be x0, x1."""
        gen = Blobs(random_state=42)
        data = gen.generate()
        assert data.feature_names == ["x0", "x1"]

    def test_target_names(self):
        """Target names should be cluster0, cluster1, etc."""
        gen = Blobs(n_clusters=3, random_state=42)
        data = gen.generate()
        assert data.target_names == ["cluster0", "cluster1", "cluster2"]

    def test_callable_return_X_y(self):
        """Callable with return_X_y=True should return tuple."""
        gen = Blobs(n_samples=10, random_state=42)
        X, y = gen(return_X_y=True)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)


class TestMoons:
    """Tests for the Two Moons clustering generator."""

    def test_default_output_shape(self):
        """Default Moons should produce (100, 2) X and (100,) y."""
        gen = Moons(random_state=42)
        data = gen.generate()
        assert data.X.shape == (100, 2)
        assert data.y.shape == (100,)

    def test_custom_n_samples(self):
        """n_samples should control number of output rows."""
        gen = Moons(n_samples=500, random_state=42)
        data = gen.generate()
        assert data.n_samples == 500

    def test_always_2d(self):
        """Moons always generates 2 features."""
        gen = Moons(random_state=42)
        data = gen.generate()
        assert data.n_features == 2

    def test_binary_labels(self):
        """Moons should produce exactly labels 0 and 1."""
        gen = Moons(n_samples=200, random_state=42)
        data = gen.generate()
        assert set(data.y) == {0, 1}

    def test_balanced_classes(self):
        """Classes should be approximately balanced."""
        gen = Moons(n_samples=200, random_state=42)
        data = gen.generate()
        count_0 = np.sum(data.y == 0)
        count_1 = np.sum(data.y == 1)
        assert count_0 == 100
        assert count_1 == 100

    def test_noise_parameter(self):
        """Noise should add Gaussian perturbation to the data."""
        gen_clean = Moons(n_samples=100, noise=0.0, shuffle=False, random_state=42)
        gen_noisy = Moons(n_samples=100, noise=0.5, shuffle=False, random_state=42)
        data_clean = gen_clean.generate()
        data_noisy = gen_noisy.generate()
        # Noisy data should differ from clean
        assert not np.array_equal(data_clean.X, data_noisy.X)

    def test_random_state_reproducibility(self):
        """Same random_state should produce identical data."""
        gen1 = Moons(n_samples=50, random_state=33)
        gen2 = Moons(n_samples=50, random_state=33)
        data1 = gen1.generate()
        data2 = gen2.generate()
        np.testing.assert_array_equal(data1.X, data2.X)
        np.testing.assert_array_equal(data1.y, data2.y)

    def test_feature_names(self):
        """Feature names should be x0, x1."""
        gen = Moons(random_state=42)
        data = gen.generate()
        assert data.feature_names == ['x0', 'x1']

    def test_target_names(self):
        """Target names should be moon0, moon1."""
        gen = Moons(random_state=42)
        data = gen.generate()
        assert data.target_names == ['moon0', 'moon1']


class TestCircles:
    """Tests for the Concentric Circles clustering generator."""

    def test_default_output_shape(self):
        """Default Circles should produce (100, 2) X and (100,) y."""
        gen = Circles(random_state=42)
        data = gen.generate()
        assert data.X.shape == (100, 2)
        assert data.y.shape == (100,)

    def test_custom_n_samples(self):
        """n_samples should control output size."""
        gen = Circles(n_samples=400, random_state=42)
        data = gen.generate()
        assert data.n_samples == 400

    def test_always_2d(self):
        """Circles always generates 2 features."""
        gen = Circles(random_state=42)
        data = gen.generate()
        assert data.n_features == 2

    def test_binary_labels(self):
        """Circles should produce labels 0 (outer) and 1 (inner)."""
        gen = Circles(n_samples=200, random_state=42)
        data = gen.generate()
        assert set(data.y) == {0, 1}

    def test_factor_affects_inner_radius(self):
        """Smaller factor should produce a smaller inner circle."""
        gen_small = Circles(n_samples=200, factor=0.1, noise=0.0,
                            shuffle=False, random_state=42)
        gen_large = Circles(n_samples=200, factor=0.8, noise=0.0,
                            shuffle=False, random_state=42)
        data_small = gen_small.generate()
        data_large = gen_large.generate()

        # Inner circle points (label=1) are in second half since shuffle=False
        inner_small = data_small.X[data_small.y == 1]
        inner_large = data_large.X[data_large.y == 1]

        # Radius of inner points should differ
        radii_small = np.sqrt(np.sum(inner_small ** 2, axis=1))
        radii_large = np.sqrt(np.sum(inner_large ** 2, axis=1))
        assert np.mean(radii_small) < np.mean(radii_large)

    def test_noise_parameter(self):
        """Noise should add perturbation to the circles."""
        gen_clean = Circles(n_samples=100, noise=0.0, shuffle=False, random_state=42)
        gen_noisy = Circles(n_samples=100, noise=0.5, shuffle=False, random_state=42)
        data_clean = gen_clean.generate()
        data_noisy = gen_noisy.generate()
        assert not np.array_equal(data_clean.X, data_noisy.X)

    def test_random_state_reproducibility(self):
        """Same random_state should produce identical data."""
        gen1 = Circles(n_samples=50, random_state=44)
        gen2 = Circles(n_samples=50, random_state=44)
        data1 = gen1.generate()
        data2 = gen2.generate()
        np.testing.assert_array_equal(data1.X, data2.X)
        np.testing.assert_array_equal(data1.y, data2.y)

    def test_target_names(self):
        """Target names should be outer and inner."""
        gen = Circles(random_state=42)
        data = gen.generate()
        assert data.target_names == ['outer', 'inner']


class TestSwissRoll:
    """Tests for the Swiss Roll clustering generator."""

    def test_default_output_shape(self):
        """Default SwissRoll should produce (100, 3) X and (100,) y."""
        gen = SwissRoll(random_state=42)
        data = gen.generate()
        assert data.X.shape == (100, 3)
        assert data.y.shape == (100,)

    def test_custom_n_samples(self):
        """n_samples should control output size."""
        gen = SwissRoll(n_samples=500, random_state=42)
        data = gen.generate()
        assert data.n_samples == 500

    def test_always_3d(self):
        """Swiss Roll always generates 3 features."""
        gen = SwissRoll(random_state=42)
        data = gen.generate()
        assert data.n_features == 3

    def test_continuous_y(self):
        """y should be continuous (position along the roll), not cluster labels."""
        gen = SwissRoll(n_samples=200, random_state=42)
        data = gen.generate()
        assert data.y.dtype in [np.float64, np.float32]
        assert len(set(data.y)) > 10

    def test_hole_parameter(self):
        """hole=True should change the range of the roll parameter t."""
        gen_no_hole = SwissRoll(n_samples=200, hole=False, random_state=42)
        gen_hole = SwissRoll(n_samples=200, hole=True, random_state=42)
        data_no_hole = gen_no_hole.generate()
        data_hole = gen_hole.generate()
        # The y values (which represent t) should have different ranges
        assert not np.array_equal(data_no_hole.y, data_hole.y)

    def test_noise_parameter(self):
        """noise > 0 should add perturbation to X."""
        gen_clean = SwissRoll(n_samples=100, noise=0.0, random_state=42)
        gen_noisy = SwissRoll(n_samples=100, noise=1.0, random_state=42)
        data_clean = gen_clean.generate()
        data_noisy = gen_noisy.generate()
        assert not np.array_equal(data_clean.X, data_noisy.X)

    def test_random_state_reproducibility(self):
        """Same random_state should produce identical data."""
        gen1 = SwissRoll(n_samples=50, random_state=66)
        gen2 = SwissRoll(n_samples=50, random_state=66)
        data1 = gen1.generate()
        data2 = gen2.generate()
        np.testing.assert_array_equal(data1.X, data2.X)
        np.testing.assert_array_equal(data1.y, data2.y)

    def test_feature_names(self):
        """Feature names should be x, y, z."""
        gen = SwissRoll(random_state=42)
        data = gen.generate()
        assert data.feature_names == ['x', 'y', 'z']
