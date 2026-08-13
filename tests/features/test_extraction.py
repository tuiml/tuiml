"""Feature extraction transformers.

Merged from: test_extraction_pca.py, test_extraction_random_projection.py
"""

import numpy as np
import pytest
from tuiml.features.extraction import PCAExtractor
from tuiml.features.extraction import (
    RandomProjectionExtractor,
    SparseRandomProjectionExtractor,
)


# --------------------------------------------------------------------------
# Tests for PCAExtractor.
# --------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Create sample data for PCA testing."""
    np.random.seed(42)
    # Create data with known structure: first 2 components capture most variance
    X = np.random.randn(100, 5)
    # Add correlated features to create a clear low-rank structure
    X[:, 3] = X[:, 0] * 2 + np.random.randn(100) * 0.1
    X[:, 4] = X[:, 1] * 1.5 + np.random.randn(100) * 0.1
    return X


class TestPCAExtractorInit:

    def test_default_init(self):
        pca = PCAExtractor()
        assert pca.n_components is None
        assert pca.center is True
        assert pca.whiten is False

    def test_custom_init(self):
        pca = PCAExtractor(n_components=3, center=False, whiten=True)
        assert pca.n_components == 3
        assert pca.center is False
        assert pca.whiten is True

    def test_attributes_none_before_fit(self):
        pca = PCAExtractor()
        assert pca.components_ is None
        assert pca.explained_variance_ is None
        assert pca.n_components_ is None


class TestPCAExtractorFitTransform:

    def test_fit_transform_reduces_dims(self, sample_data):
        pca = PCAExtractor(n_components=2)
        X_new = pca.fit_transform(sample_data)
        assert X_new.shape == (100, 2)

    def test_fit_then_transform_matches_fit_transform(self, sample_data):
        pca1 = PCAExtractor(n_components=3)
        X1 = pca1.fit_transform(sample_data)

        pca2 = PCAExtractor(n_components=3)
        pca2.fit(sample_data)
        X2 = pca2.transform(sample_data)

        np.testing.assert_allclose(X1, X2, atol=1e-10)

    def test_n_components_none_keeps_all(self, sample_data):
        pca = PCAExtractor(n_components=None)
        X_new = pca.fit_transform(sample_data)
        assert X_new.shape[1] == min(sample_data.shape)

    def test_n_components_exceeds_features(self, sample_data):
        pca = PCAExtractor(n_components=50)
        X_new = pca.fit_transform(sample_data)
        assert X_new.shape[1] == min(sample_data.shape)


class TestPCAExtractorVariance:

    def test_explained_variance_ratio_sums_to_1(self, sample_data):
        pca = PCAExtractor(n_components=None)
        pca.fit(sample_data)
        total = np.sum(pca.explained_variance_ratio_)
        np.testing.assert_allclose(total, 1.0, atol=1e-10)

    def test_explained_variance_decreasing(self, sample_data):
        pca = PCAExtractor(n_components=None)
        pca.fit(sample_data)
        # Each subsequent component should explain less or equal variance
        ratios = pca.explained_variance_ratio_
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1] - 1e-10

    def test_n_components_float(self, sample_data):
        """When n_components is a float, it should select components to explain that
        proportion of variance."""
        pca = PCAExtractor(n_components=0.95)
        pca.fit(sample_data)
        assert pca.n_components_ >= 1
        assert pca.n_components_ <= sample_data.shape[1]
        # The explained variance should cover at least 95%
        actual_explained = np.sum(pca.explained_variance_ratio_)
        assert actual_explained >= 0.95 - 1e-5


class TestPCAExtractorInverse:

    def test_inverse_transform_reconstruction(self, sample_data):
        pca = PCAExtractor(n_components=None)
        X_transformed = pca.fit_transform(sample_data)
        X_reconstructed = pca.inverse_transform(X_transformed)
        # With all components, reconstruction should be near-perfect
        np.testing.assert_allclose(X_reconstructed, sample_data, atol=1e-8)

    def test_inverse_transform_partial_reconstruction(self, sample_data):
        pca = PCAExtractor(n_components=2)
        X_transformed = pca.fit_transform(sample_data)
        X_reconstructed = pca.inverse_transform(X_transformed)
        # With fewer components, reconstruction is approximate
        assert X_reconstructed.shape == sample_data.shape
        # Should not be exact
        error = np.mean((X_reconstructed - sample_data) ** 2)
        assert error > 0


class TestPCAExtractorWhiten:

    def test_whiten_produces_uniform_variance(self, sample_data):
        pca = PCAExtractor(n_components=3, whiten=True)
        X_new = pca.fit_transform(sample_data)
        # Whitened components should all have the same variance
        # (the PCA implementation divides by singular_values_, giving var = 1/(n-1))
        variances = np.var(X_new, axis=0)
        # All components should have equal variance after whitening
        np.testing.assert_allclose(variances, variances[0], atol=1e-10)
        # Variance should be positive
        assert variances[0] > 0

    def test_whiten_inverse_transform(self, sample_data):
        pca = PCAExtractor(n_components=None, whiten=True)
        X_transformed = pca.fit_transform(sample_data)
        X_reconstructed = pca.inverse_transform(X_transformed)
        np.testing.assert_allclose(X_reconstructed, sample_data, atol=1e-8)


class TestPCAExtractorFeatureNames:

    def test_get_feature_names(self, sample_data):
        pca = PCAExtractor(n_components=3)
        pca.fit(sample_data)
        names = pca.get_feature_names_out()
        assert len(names) == 3
        assert names[0] == "PC1"
        assert names[1] == "PC2"
        assert names[2] == "PC3"

    def test_get_feature_names_before_fit_raises(self):
        pca = PCAExtractor(n_components=3)
        with pytest.raises(RuntimeError):
            pca.get_feature_names_out()


class TestPCAExtractorCovariance:

    def test_get_covariance_shape(self, sample_data):
        pca = PCAExtractor(n_components=3)
        pca.fit(sample_data)
        cov = pca.get_covariance()
        assert cov.shape == (sample_data.shape[1], sample_data.shape[1])

    def test_get_precision_shape(self, sample_data):
        pca = PCAExtractor(n_components=3)
        pca.fit(sample_data)
        precision = pca.get_precision()
        assert precision.shape == (sample_data.shape[1], sample_data.shape[1])


class TestPCAExtractorErrors:

    def test_inverse_transform_before_fit_raises(self):
        pca = PCAExtractor(n_components=2)
        X = np.random.randn(10, 2)
        with pytest.raises(RuntimeError):
            pca.inverse_transform(X)


class TestPCAExtractorSchema:

    def test_get_parameter_schema(self):
        schema = PCAExtractor.get_parameter_schema()
        assert "n_components" in schema
        assert "center" in schema
        assert "whiten" in schema

    def test_repr(self):
        pca = PCAExtractor(n_components=3, center=True, whiten=False)
        repr_str = repr(pca)
        assert "PCAExtractor" in repr_str
        assert "n_components=3" in repr_str


# --------------------------------------------------------------------------
# Tests for RandomProjectionExtractor and SparseRandomProjectionExtractor.
# --------------------------------------------------------------------------

@pytest.fixture
def projection_sample_data():
    """Create sample data for random projection testing."""
    np.random.seed(42)
    return np.random.randn(100, 50)


class TestRandomProjectionExtractorInit:

    def test_default_init(self):
        rp = RandomProjectionExtractor()
        assert rp.n_components == 10
        assert rp.distribution == "gaussian"
        assert rp.random_state is None

    def test_custom_init(self):
        rp = RandomProjectionExtractor(
            n_components=20, distribution="sparse", random_state=42
        )
        assert rp.n_components == 20
        assert rp.distribution == "sparse"
        assert rp.random_state == 42


class TestRandomProjectionExtractorFit:

    def test_fit_gaussian(self, projection_sample_data):
        rp = RandomProjectionExtractor(n_components=10, distribution="gaussian", random_state=42)
        rp.fit(projection_sample_data)
        assert rp.n_components_ == 10
        assert rp.components_.shape == (10, 50)

    def test_fit_sparse(self, projection_sample_data):
        rp = RandomProjectionExtractor(n_components=10, distribution="sparse", random_state=42)
        rp.fit(projection_sample_data)
        assert rp.n_components_ == 10

    def test_fit_rademacher(self, projection_sample_data):
        rp = RandomProjectionExtractor(n_components=10, distribution="rademacher", random_state=42)
        rp.fit(projection_sample_data)
        # rademacher should only have -1 and 1 values
        assert set(np.unique(rp.components_)).issubset({-1.0, 1.0})

    def test_fit_auto_components(self, projection_sample_data):
        rp = RandomProjectionExtractor(n_components="auto", random_state=42)
        rp.fit(projection_sample_data)
        assert rp.n_components_ >= 1

    def test_fit_float_components(self, projection_sample_data):
        rp = RandomProjectionExtractor(n_components=0.2, random_state=42)
        rp.fit(projection_sample_data)
        # 20% of 50 = 10
        assert rp.n_components_ == 10


class TestRandomProjectionExtractorTransform:

    def test_transform_output_shape(self, projection_sample_data):
        rp = RandomProjectionExtractor(n_components=15, random_state=42)
        X_new = rp.fit_transform(projection_sample_data)
        assert X_new.shape == (100, 15)

    def test_transform_wrong_features_raises(self, projection_sample_data):
        rp = RandomProjectionExtractor(n_components=10, random_state=42)
        rp.fit(projection_sample_data)
        X_wrong = np.random.randn(10, 30)  # wrong number of features
        with pytest.raises(ValueError, match="features"):
            rp.transform(X_wrong)

    def test_output_dimensionality(self, projection_sample_data):
        """Verify output has the specified number of dimensions."""
        for k in [2, 5, 20, 40]:
            rp = RandomProjectionExtractor(n_components=k, random_state=42)
            X_new = rp.fit_transform(projection_sample_data)
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

    def test_same_random_state(self, projection_sample_data):
        rp1 = RandomProjectionExtractor(n_components=10, random_state=42)
        rp2 = RandomProjectionExtractor(n_components=10, random_state=42)

        X1 = rp1.fit_transform(projection_sample_data)
        X2 = rp2.fit_transform(projection_sample_data)

        np.testing.assert_array_equal(X1, X2)


class TestRandomProjectionExtractorMisc:

    def test_invalid_distribution_raises(self, projection_sample_data):
        rp = RandomProjectionExtractor(n_components=10, distribution="invalid")
        with pytest.raises(ValueError, match="Unknown distribution"):
            rp.fit(projection_sample_data)

    def test_get_feature_names_out(self, projection_sample_data):
        rp = RandomProjectionExtractor(n_components=3, random_state=42)
        rp.fit(projection_sample_data)
        names = rp.get_feature_names_out()
        assert len(names) == 3
        assert names[0] == "rp0"
        assert names[1] == "rp1"
        assert names[2] == "rp2"

    def test_inverse_transform(self, projection_sample_data):
        rp = RandomProjectionExtractor(n_components=10, random_state=42)
        X_proj = rp.fit_transform(projection_sample_data)
        X_recon = rp.inverse_transform(X_proj)
        # Reconstruction is approximate
        assert X_recon.shape == projection_sample_data.shape

    def test_get_parameter_schema(self):
        schema = RandomProjectionExtractor.get_parameter_schema()
        assert "n_components" in schema
        assert "distribution" in schema
        assert "random_state" in schema


class TestSparseRandomProjectionExtractor:

    def test_default_init(self):
        srp = SparseRandomProjectionExtractor()
        assert srp.n_components == 10
        assert srp.distribution == "sparse"

    def test_fit_transform(self, projection_sample_data):
        srp = SparseRandomProjectionExtractor(n_components=10, random_state=42)
        X_new = srp.fit_transform(projection_sample_data)
        assert X_new.shape == (100, 10)

    def test_get_parameter_schema(self):
        schema = SparseRandomProjectionExtractor.get_parameter_schema()
        assert "n_components" in schema
        assert "density" in schema
        assert "random_state" in schema

    def test_reproducibility(self, projection_sample_data):
        srp1 = SparseRandomProjectionExtractor(n_components=10, random_state=42)
        srp2 = SparseRandomProjectionExtractor(n_components=10, random_state=42)

        X1 = srp1.fit_transform(projection_sample_data)
        X2 = srp2.fit_transform(projection_sample_data)

        np.testing.assert_array_equal(X1, X2)
