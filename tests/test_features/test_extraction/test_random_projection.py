"""Tests for RandomProjectionExtractor and SparseRandomProjectionExtractor."""

import numpy as np
import pytest

from tuiml.features.extraction import (
    RandomProjectionExtractor,
    SparseRandomProjectionExtractor,
)


@pytest.fixture
def sample_data():
    """Create sample data for random projection testing."""
    np.random.seed(42)
    return np.random.randn(100, 50)


# ---- RandomProjectionExtractor ----

class TestRandomProjectionExtractorInit:

    def test_default_init(self):
        rp = RandomProjectionExtractor()
        assert rp.n_components == 10
        assert rp.distribution == "gaussian"
        assert rp.random_state is None

    def test_custom_init(self):
        rp = RandomProjectionExtractor(
            n_components=20, distribution="sparse1", random_state=42
        )
        assert rp.n_components == 20
        assert rp.distribution == "sparse1"
        assert rp.random_state == 42


class TestRandomProjectionExtractorFit:

    def test_fit_gaussian(self, sample_data):
        rp = RandomProjectionExtractor(n_components=10, distribution="gaussian", random_state=42)
        rp.fit(sample_data)
        assert rp.n_components_ == 10
        assert rp.components_.shape == (10, 50)

    def test_fit_sparse1(self, sample_data):
        rp = RandomProjectionExtractor(n_components=10, distribution="sparse1", random_state=42)
        rp.fit(sample_data)
        assert rp.n_components_ == 10

    def test_fit_sparse2(self, sample_data):
        rp = RandomProjectionExtractor(n_components=10, distribution="sparse2", random_state=42)
        rp.fit(sample_data)
        # sparse2 should only have -1 and 1 values
        assert set(np.unique(rp.components_)).issubset({-1.0, 1.0})

    def test_fit_auto_components(self, sample_data):
        rp = RandomProjectionExtractor(n_components="auto", random_state=42)
        rp.fit(sample_data)
        assert rp.n_components_ >= 1

    def test_fit_float_components(self, sample_data):
        rp = RandomProjectionExtractor(n_components=0.2, random_state=42)
        rp.fit(sample_data)
        # 20% of 50 = 10
        assert rp.n_components_ == 10


class TestRandomProjectionExtractorTransform:

    def test_transform_output_shape(self, sample_data):
        rp = RandomProjectionExtractor(n_components=15, random_state=42)
        X_new = rp.fit_transform(sample_data)
        assert X_new.shape == (100, 15)

    def test_transform_before_fit_raises(self, sample_data):
        rp = RandomProjectionExtractor(n_components=10)
        with pytest.raises(RuntimeError):
            rp.transform(sample_data)

    def test_transform_wrong_features_raises(self, sample_data):
        rp = RandomProjectionExtractor(n_components=10, random_state=42)
        rp.fit(sample_data)
        X_wrong = np.random.randn(10, 30)  # wrong number of features
        with pytest.raises(ValueError, match="features"):
            rp.transform(X_wrong)

    def test_output_dimensionality(self, sample_data):
        """Verify output has the specified number of dimensions."""
        for k in [2, 5, 20, 40]:
            rp = RandomProjectionExtractor(n_components=k, random_state=42)
            X_new = rp.fit_transform(sample_data)
            assert X_new.shape[1] == k


class TestRandomProjectionExtractorDistancePreservation:

    def test_pairwise_distances_approximately_preserved(self):
        """Random projection should approximately preserve pairwise distances."""
        np.random.seed(42)
        X = np.random.randn(20, 100)

        rp = RandomProjectionExtractor(n_components=50, random_state=42)
        X_proj = rp.fit_transform(X)

        # Compute pairwise distances in original and projected space
        from itertools import combinations
        original_dists = []
        projected_dists = []
        for i, j in combinations(range(20), 2):
            original_dists.append(np.linalg.norm(X[i] - X[j]))
            projected_dists.append(np.linalg.norm(X_proj[i] - X_proj[j]))

        original_dists = np.array(original_dists)
        projected_dists = np.array(projected_dists)

        # Distances should be correlated (not exact, but correlated)
        correlation = np.corrcoef(original_dists, projected_dists)[0, 1]
        assert correlation > 0.5


class TestRandomProjectionExtractorReproducibility:

    def test_same_random_state(self, sample_data):
        rp1 = RandomProjectionExtractor(n_components=10, random_state=42)
        rp2 = RandomProjectionExtractor(n_components=10, random_state=42)

        X1 = rp1.fit_transform(sample_data)
        X2 = rp2.fit_transform(sample_data)

        np.testing.assert_array_equal(X1, X2)


class TestRandomProjectionExtractorMisc:

    def test_invalid_distribution_raises(self, sample_data):
        rp = RandomProjectionExtractor(n_components=10, distribution="invalid")
        with pytest.raises(ValueError, match="Unknown distribution"):
            rp.fit(sample_data)

    def test_get_feature_names_out(self, sample_data):
        rp = RandomProjectionExtractor(n_components=3, random_state=42)
        rp.fit(sample_data)
        names = rp.get_feature_names_out()
        assert len(names) == 3
        assert names[0] == "rp0"
        assert names[1] == "rp1"
        assert names[2] == "rp2"

    def test_inverse_transform(self, sample_data):
        rp = RandomProjectionExtractor(n_components=10, random_state=42)
        X_proj = rp.fit_transform(sample_data)
        X_recon = rp.inverse_transform(X_proj)
        # Reconstruction is approximate
        assert X_recon.shape == sample_data.shape

    def test_get_parameter_schema(self):
        schema = RandomProjectionExtractor.get_parameter_schema()
        assert "n_components" in schema
        assert "distribution" in schema
        assert "random_state" in schema


# ---- SparseRandomProjectionExtractor ----

class TestSparseRandomProjectionExtractor:

    def test_default_init(self):
        srp = SparseRandomProjectionExtractor()
        assert srp.n_components == 10
        assert srp.distribution == "sparse1"

    def test_fit_transform(self, sample_data):
        srp = SparseRandomProjectionExtractor(n_components=10, random_state=42)
        X_new = srp.fit_transform(sample_data)
        assert X_new.shape == (100, 10)

    def test_get_parameter_schema(self):
        schema = SparseRandomProjectionExtractor.get_parameter_schema()
        assert "n_components" in schema
        assert "density" in schema
        assert "random_state" in schema

    def test_reproducibility(self, sample_data):
        srp1 = SparseRandomProjectionExtractor(n_components=10, random_state=42)
        srp2 = SparseRandomProjectionExtractor(n_components=10, random_state=42)

        X1 = srp1.fit_transform(sample_data)
        X2 = srp2.fit_transform(sample_data)

        np.testing.assert_array_equal(X1, X2)
