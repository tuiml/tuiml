"""Tests for PolynomialFeaturesGenerator and InteractionFeaturesGenerator."""

import numpy as np
import pytest

from tuiml.features.generation import (
    PolynomialFeaturesGenerator,
    InteractionFeaturesGenerator,
)


@pytest.fixture
def simple_data():
    """Create simple 2-feature data for verifiable polynomial outputs."""
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


@pytest.fixture
def larger_data():
    """Create larger data for shape tests."""
    np.random.seed(42)
    return np.random.randn(50, 4)


# ---- PolynomialFeaturesGenerator ----

class TestPolynomialFeaturesGeneratorInit:

    def test_default_init(self):
        poly = PolynomialFeaturesGenerator()
        assert poly.degree == 2
        assert poly.interaction_only is False
        assert poly.include_bias is True

    def test_custom_init(self):
        poly = PolynomialFeaturesGenerator(
            degree=3, interaction_only=True, include_bias=False
        )
        assert poly.degree == 3
        assert poly.interaction_only is True
        assert poly.include_bias is False


class TestPolynomialFeaturesGeneratorFit:

    def test_fit_stores_attributes(self, simple_data):
        poly = PolynomialFeaturesGenerator(degree=2)
        poly.fit(simple_data)
        assert poly.n_input_features_ == 2
        assert poly.n_output_features_ is not None
        assert poly.powers_ is not None

    def test_transform_before_fit_raises(self, simple_data):
        poly = PolynomialFeaturesGenerator()
        with pytest.raises(RuntimeError):
            poly.transform(simple_data)


class TestPolynomialFeaturesGeneratorTransform:

    def test_degree2_with_bias(self, simple_data):
        """For [a, b], degree 2 with bias: [1, a, b, a^2, ab, b^2]."""
        poly = PolynomialFeaturesGenerator(degree=2, include_bias=True)
        X_poly = poly.fit_transform(simple_data)
        # Expected: 1 + 2 + 3 = 6 features (bias + degree1 + degree2)
        assert X_poly.shape == (3, 6)

        # Verify first row: [1, 2] -> [1, 1, 2, 1, 2, 4]
        np.testing.assert_allclose(X_poly[0, 0], 1.0)   # bias
        np.testing.assert_allclose(X_poly[0, 1], 1.0)   # a
        np.testing.assert_allclose(X_poly[0, 2], 2.0)   # b
        np.testing.assert_allclose(X_poly[0, 3], 1.0)   # a^2
        np.testing.assert_allclose(X_poly[0, 4], 2.0)   # a*b
        np.testing.assert_allclose(X_poly[0, 5], 4.0)   # b^2

    def test_degree2_without_bias(self, simple_data):
        poly = PolynomialFeaturesGenerator(degree=2, include_bias=False)
        X_poly = poly.fit_transform(simple_data)
        # Expected: 2 + 3 = 5 features
        assert X_poly.shape == (3, 5)

    def test_interaction_only(self, simple_data):
        poly = PolynomialFeaturesGenerator(
            degree=2, interaction_only=True, include_bias=False
        )
        X_poly = poly.fit_transform(simple_data)
        # For 2 features, interaction_only degree=2: [a, b, a*b] = 3 features
        assert X_poly.shape == (3, 3)

    def test_degree3_output_count(self, larger_data):
        poly = PolynomialFeaturesGenerator(degree=3, include_bias=True)
        X_poly = poly.fit_transform(larger_data)
        assert X_poly.shape[0] == larger_data.shape[0]
        # For 4 features, degree 3 with bias: C(4+3, 3) = 35
        assert X_poly.shape[1] == 35

    def test_wrong_n_features_raises(self, simple_data):
        poly = PolynomialFeaturesGenerator(degree=2)
        poly.fit(simple_data)
        X_wrong = np.random.randn(5, 5)  # 5 features instead of 2
        with pytest.raises(ValueError, match="features"):
            poly.transform(X_wrong)


class TestPolynomialFeaturesGeneratorNames:

    def test_get_feature_names_out_default(self, simple_data):
        poly = PolynomialFeaturesGenerator(degree=2, include_bias=True)
        poly.fit(simple_data)
        names = poly.get_feature_names_out()
        assert len(names) == poly.n_output_features_
        assert names[0] == "1"  # bias term

    def test_get_feature_names_out_custom(self, simple_data):
        poly = PolynomialFeaturesGenerator(degree=2, include_bias=True)
        poly.fit(simple_data)
        names = poly.get_feature_names_out(["a", "b"])
        assert "1" in names
        assert "a" in names
        assert "b" in names
        assert "a^2" in names
        assert "a*b" in names
        assert "b^2" in names


class TestPolynomialFeaturesGeneratorSchema:

    def test_get_parameter_schema(self):
        schema = PolynomialFeaturesGenerator.get_parameter_schema()
        assert "degree" in schema
        assert "interaction_only" in schema
        assert "include_bias" in schema


# ---- InteractionFeaturesGenerator ----

class TestInteractionFeaturesGeneratorInit:

    def test_default_init(self):
        inter = InteractionFeaturesGenerator()
        assert inter.include_original is True

    def test_custom_init(self):
        inter = InteractionFeaturesGenerator(include_original=False)
        assert inter.include_original is False


class TestInteractionFeaturesGeneratorFit:

    def test_fit_stores_attributes(self, simple_data):
        inter = InteractionFeaturesGenerator()
        inter.fit(simple_data)
        assert inter.n_input_features_ == 2
        assert inter.interaction_pairs_ is not None
        # For 2 features, there's 1 interaction pair: (0, 1)
        assert len(inter.interaction_pairs_) == 1

    def test_transform_before_fit_raises(self, simple_data):
        inter = InteractionFeaturesGenerator()
        with pytest.raises(RuntimeError):
            inter.transform(simple_data)


class TestInteractionFeaturesGeneratorTransform:

    def test_include_original(self, simple_data):
        inter = InteractionFeaturesGenerator(include_original=True)
        X_new = inter.fit_transform(simple_data)
        # 2 original + 1 interaction = 3
        assert X_new.shape == (3, 3)
        # Original features should be preserved
        np.testing.assert_allclose(X_new[:, :2], simple_data)
        # Interaction: a * b
        np.testing.assert_allclose(X_new[:, 2], simple_data[:, 0] * simple_data[:, 1])

    def test_exclude_original(self, simple_data):
        inter = InteractionFeaturesGenerator(include_original=False)
        X_new = inter.fit_transform(simple_data)
        # Only 1 interaction feature
        assert X_new.shape == (3, 1)

    def test_three_features(self):
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        inter = InteractionFeaturesGenerator(include_original=True)
        X_new = inter.fit_transform(X)
        # 3 original + 3 interactions (0,1), (0,2), (1,2) = 6
        assert X_new.shape == (2, 6)

    def test_four_features_interaction_count(self, larger_data):
        inter = InteractionFeaturesGenerator(include_original=False)
        X_new = inter.fit_transform(larger_data)
        # C(4, 2) = 6 interaction features
        assert X_new.shape[1] == 6


class TestInteractionFeaturesGeneratorNames:

    def test_get_feature_names_out(self, simple_data):
        inter = InteractionFeaturesGenerator(include_original=True)
        inter.fit(simple_data)
        names = inter.get_feature_names_out(["a", "b"])
        assert "a" in names
        assert "b" in names
        assert "a*b" in names


class TestInteractionFeaturesGeneratorSchema:

    def test_get_parameter_schema(self):
        schema = InteractionFeaturesGenerator.get_parameter_schema()
        assert "include_original" in schema
