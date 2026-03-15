"""Tests for PCAExtractor."""

import numpy as np
import pytest

from tuiml.features.extraction import PCAExtractor


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

    def test_transform_before_fit_raises(self):
        pca = PCAExtractor(n_components=2)
        X = np.random.randn(10, 5)
        with pytest.raises(RuntimeError):
            pca.transform(X)

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
