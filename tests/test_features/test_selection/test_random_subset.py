"""Tests for RandomSubsetSelector and BootstrapFeaturesSelector."""

import numpy as np
import pytest

from tuiml.features.selection import RandomSubsetSelector, BootstrapFeaturesSelector


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return np.random.randn(30, 20)


# ---- RandomSubsetSelector ----

class TestRandomSubsetSelectorInit:

    def test_default_init(self):
        selector = RandomSubsetSelector()
        assert selector.n_features == 0.5
        assert selector.invert is False
        assert selector.random_state is None

    def test_custom_init(self):
        selector = RandomSubsetSelector(n_features=5, invert=True, random_state=42)
        assert selector.n_features == 5
        assert selector.invert is True
        assert selector.random_state == 42


class TestRandomSubsetSelectorFit:

    def test_fit_fraction(self, sample_data):
        selector = RandomSubsetSelector(n_features=0.5, random_state=42)
        selector.fit(sample_data)
        # 50% of 20 = 10
        assert selector.n_features_selected_ == 10
        assert len(selector.selected_features_) == 10

    def test_fit_absolute(self, sample_data):
        selector = RandomSubsetSelector(n_features=7, random_state=42)
        selector.fit(sample_data)
        assert selector.n_features_selected_ == 7

    def test_transform_output_shape(self, sample_data):
        selector = RandomSubsetSelector(n_features=0.3, random_state=42)
        X_new = selector.fit_transform(sample_data)
        expected_n = max(1, int(round(20 * 0.3)))
        assert X_new.shape == (30, expected_n)

    def test_invert_selection(self, sample_data):
        selector_normal = RandomSubsetSelector(n_features=5, random_state=42, invert=False)
        selector_invert = RandomSubsetSelector(n_features=5, random_state=42, invert=True)

        selector_normal.fit(sample_data)
        selector_invert.fit(sample_data)

        # Invert should select the complement
        assert selector_normal.n_features_selected_ + selector_invert.n_features_selected_ == 20

    def test_unsupervised_no_y_needed(self, sample_data):
        selector = RandomSubsetSelector(n_features=5, random_state=42)
        # Should not raise even without y
        selector.fit(sample_data)
        assert selector.n_features_selected_ == 5


class TestRandomSubsetSelectorReproducibility:

    def test_same_random_state_same_result(self, sample_data):
        selector1 = RandomSubsetSelector(n_features=5, random_state=42)
        selector2 = RandomSubsetSelector(n_features=5, random_state=42)

        X1 = selector1.fit_transform(sample_data)
        X2 = selector2.fit_transform(sample_data)

        np.testing.assert_array_equal(X1, X2)

    def test_different_random_state_different_result(self, sample_data):
        selector1 = RandomSubsetSelector(n_features=5, random_state=42)
        selector2 = RandomSubsetSelector(n_features=5, random_state=99)

        selector1.fit(sample_data)
        selector2.fit(sample_data)

        # Very likely to be different (though not guaranteed for small n)
        assert not np.array_equal(
            selector1.selected_features_, selector2.selected_features_
        )


class TestRandomSubsetSelectorTransform:

    def test_transform_before_fit_raises(self, sample_data):
        selector = RandomSubsetSelector(n_features=5)
        with pytest.raises(RuntimeError):
            selector.transform(sample_data)

    def test_transform_wrong_n_features_raises(self, sample_data):
        selector = RandomSubsetSelector(n_features=5, random_state=42)
        selector.fit(sample_data)
        X_wrong = np.random.randn(10, 15)  # wrong number of features
        with pytest.raises(ValueError, match="features"):
            selector.transform(X_wrong)


class TestRandomSubsetSelectorSchema:

    def test_get_parameter_schema(self):
        schema = RandomSubsetSelector.get_parameter_schema()
        assert "n_features" in schema
        assert "invert" in schema
        assert "random_state" in schema


# ---- BootstrapFeaturesSelector ----

class TestBootstrapFeaturesSelectorInit:

    def test_default_init(self):
        selector = BootstrapFeaturesSelector()
        assert selector.n_features == "sqrt"
        assert selector.random_state is None

    def test_custom_init(self):
        selector = BootstrapFeaturesSelector(n_features="log2", random_state=42)
        assert selector.n_features == "log2"
        assert selector.random_state == 42


class TestBootstrapFeaturesSelectorFit:

    def test_sqrt_selection(self, sample_data):
        selector = BootstrapFeaturesSelector(n_features="sqrt", random_state=42)
        selector.fit(sample_data)
        # sqrt(20) ~ 4, but bootstrap may produce fewer unique indices
        assert selector.n_features_selected_ >= 1
        assert selector.n_features_selected_ <= 20

    def test_log2_selection(self, sample_data):
        selector = BootstrapFeaturesSelector(n_features="log2", random_state=42)
        selector.fit(sample_data)
        assert selector.n_features_selected_ >= 1

    def test_fraction_selection(self, sample_data):
        selector = BootstrapFeaturesSelector(n_features=0.5, random_state=42)
        selector.fit(sample_data)
        assert selector.n_features_selected_ >= 1

    def test_absolute_selection(self, sample_data):
        selector = BootstrapFeaturesSelector(n_features=8, random_state=42)
        selector.fit(sample_data)
        # Bootstrap samples with replacement then takes unique, so may be fewer
        assert selector.n_features_selected_ >= 1
        assert selector.n_features_selected_ <= 8

    def test_transform_output_shape(self, sample_data):
        selector = BootstrapFeaturesSelector(n_features="sqrt", random_state=42)
        X_new = selector.fit_transform(sample_data)
        assert X_new.shape[0] == sample_data.shape[0]
        assert X_new.shape[1] == selector.n_features_selected_

    def test_reproducibility(self, sample_data):
        selector1 = BootstrapFeaturesSelector(n_features="sqrt", random_state=42)
        selector2 = BootstrapFeaturesSelector(n_features="sqrt", random_state=42)

        X1 = selector1.fit_transform(sample_data)
        X2 = selector2.fit_transform(sample_data)

        np.testing.assert_array_equal(X1, X2)

    def test_invalid_n_features_string_raises(self, sample_data):
        selector = BootstrapFeaturesSelector(n_features="invalid")
        with pytest.raises(ValueError, match="Unknown n_features string"):
            selector.fit(sample_data)


class TestBootstrapFeaturesSelectorSchema:

    def test_get_parameter_schema(self):
        schema = BootstrapFeaturesSelector.get_parameter_schema()
        assert "n_features" in schema
        assert "random_state" in schema
