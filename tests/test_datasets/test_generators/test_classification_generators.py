"""Tests for classification data generators."""

import numpy as np
import pytest

from tuiml.datasets.generators.classification.hyperplane import Hyperplane
from tuiml.datasets.generators.classification.agrawal import Agrawal
from tuiml.datasets.generators.classification.led import LED
from tuiml.datasets.generators.classification.random_rbf import RandomRBF
from tuiml.base.generators import GeneratedData


# ===================================================================
# Hyperplane generator tests
# ===================================================================

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


# ===================================================================
# Agrawal generator tests
# ===================================================================

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


# ===================================================================
# LED generator tests
# ===================================================================

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


# ===================================================================
# RandomRBF generator tests
# ===================================================================

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
