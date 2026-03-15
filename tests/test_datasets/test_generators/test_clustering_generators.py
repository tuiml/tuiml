"""Tests for clustering data generators."""

import numpy as np
import pytest

from tuiml.datasets.generators.clustering.blobs import Blobs
from tuiml.datasets.generators.clustering.moons import Moons
from tuiml.datasets.generators.clustering.circles import Circles
from tuiml.datasets.generators.clustering.swiss_roll import SwissRoll
from tuiml.base.generators import GeneratedData


# ===================================================================
# Blobs generator tests
# ===================================================================

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


# ===================================================================
# Moons generator tests
# ===================================================================

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


# ===================================================================
# Circles generator tests
# ===================================================================

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


# ===================================================================
# SwissRoll generator tests
# ===================================================================

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
