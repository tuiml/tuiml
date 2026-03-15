"""Tests for MathematicalFeaturesGenerator and BinningFeaturesGenerator."""

import numpy as np
import pytest

from tuiml.features.generation import (
    MathematicalFeaturesGenerator,
    BinningFeaturesGenerator,
)


@pytest.fixture
def positive_data():
    """Create strictly positive data suitable for log/sqrt transforms."""
    np.random.seed(42)
    return np.abs(np.random.randn(30, 3)) + 0.1


@pytest.fixture
def general_data():
    """Create general-purpose data."""
    np.random.seed(42)
    return np.random.randn(50, 4)


# ---- MathematicalFeaturesGenerator ----

class TestMathematicalFeaturesGeneratorInit:

    def test_default_init(self):
        gen = MathematicalFeaturesGenerator()
        assert gen.transformations == ["log1p", "sqrt", "square"]
        assert gen.include_original is True
        assert gen.handle_invalid == "nan"

    def test_custom_init(self):
        gen = MathematicalFeaturesGenerator(
            transformations=["exp", "sin"],
            include_original=False,
            handle_invalid="clip"
        )
        assert gen.transformations == ["exp", "sin"]
        assert gen.include_original is False
        assert gen.handle_invalid == "clip"


class TestMathematicalFeaturesGeneratorFit:

    def test_fit_stores_attributes(self, positive_data):
        gen = MathematicalFeaturesGenerator(transformations=["sqrt", "square"])
        gen.fit(positive_data)
        assert gen.n_input_features_ == 3
        # 3 original + 3*2 transforms = 9
        assert gen.n_output_features_ == 9

    def test_fit_without_original(self, positive_data):
        gen = MathematicalFeaturesGenerator(
            transformations=["sqrt", "square"],
            include_original=False
        )
        gen.fit(positive_data)
        # 3 * 2 = 6 (no originals)
        assert gen.n_output_features_ == 6

    def test_invalid_transformation_raises(self, positive_data):
        gen = MathematicalFeaturesGenerator(transformations=["nonexistent_func"])
        with pytest.raises(ValueError, match="Unknown transformation"):
            gen.fit(positive_data)

    def test_transform_before_fit_raises(self, positive_data):
        gen = MathematicalFeaturesGenerator()
        with pytest.raises(RuntimeError):
            gen.transform(positive_data)


class TestMathematicalFeaturesGeneratorTransform:

    def test_transform_output_shape(self, positive_data):
        gen = MathematicalFeaturesGenerator(
            transformations=["sqrt", "square"],
            include_original=True
        )
        X_new = gen.fit_transform(positive_data)
        # 3 original + 2*3 = 9
        assert X_new.shape == (30, 9)

    def test_transform_preserves_originals(self, positive_data):
        gen = MathematicalFeaturesGenerator(
            transformations=["sqrt"],
            include_original=True
        )
        X_new = gen.fit_transform(positive_data)
        # First 3 columns should be original
        np.testing.assert_allclose(X_new[:, :3], positive_data)

    def test_sqrt_values_correct(self, positive_data):
        gen = MathematicalFeaturesGenerator(
            transformations=["sqrt"],
            include_original=True
        )
        X_new = gen.fit_transform(positive_data)
        # Columns 3,4,5 should be sqrt of columns 0,1,2
        expected_sqrt = np.sqrt(positive_data)
        np.testing.assert_allclose(X_new[:, 3:], expected_sqrt, atol=1e-10)

    def test_square_values_correct(self, positive_data):
        gen = MathematicalFeaturesGenerator(
            transformations=["square"],
            include_original=False
        )
        X_new = gen.fit_transform(positive_data)
        expected_square = np.square(positive_data)
        np.testing.assert_allclose(X_new, expected_square, atol=1e-10)

    def test_handle_invalid_nan(self, general_data):
        """Log of negative numbers should produce NaN when handle_invalid='nan'."""
        gen = MathematicalFeaturesGenerator(
            transformations=["log"],
            include_original=False,
            handle_invalid="nan"
        )
        X_new = gen.fit_transform(general_data)
        # Some values in general_data are negative, so log produces NaN
        has_nan = np.any(np.isnan(X_new))
        assert has_nan

    def test_handle_invalid_clip(self, general_data):
        """With clip, log of negative numbers should be handled by clipping."""
        gen = MathematicalFeaturesGenerator(
            transformations=["log"],
            include_original=False,
            handle_invalid="clip"
        )
        X_new = gen.fit_transform(general_data)
        # After clipping, there should be no NaN
        assert not np.any(np.isnan(X_new))

    def test_handle_invalid_raise(self, general_data):
        gen = MathematicalFeaturesGenerator(
            transformations=["log"],
            include_original=False,
            handle_invalid="raise"
        )
        gen.fit(general_data)
        with pytest.raises(ValueError, match="Invalid values"):
            gen.transform(general_data)

    def test_wrong_n_features_raises(self, positive_data):
        gen = MathematicalFeaturesGenerator(transformations=["sqrt"])
        gen.fit(positive_data)
        X_wrong = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="features"):
            gen.transform(X_wrong)


class TestMathematicalFeaturesGeneratorAllTransforms:

    def test_all_supported_transforms(self, positive_data):
        """Test that all named transforms can be applied without errors."""
        all_transforms = [
            "log", "log1p", "log10", "log2", "sqrt", "cbrt",
            "exp", "expm1", "square", "abs", "sin", "cos", "tan",
            "tanh", "sigmoid"
        ]
        gen = MathematicalFeaturesGenerator(
            transformations=all_transforms,
            include_original=False,
            handle_invalid="nan"
        )
        X_new = gen.fit_transform(positive_data)
        assert X_new.shape == (30, 3 * len(all_transforms))


class TestMathematicalFeaturesGeneratorNames:

    def test_get_feature_names_out(self, positive_data):
        gen = MathematicalFeaturesGenerator(
            transformations=["sqrt", "square"],
            include_original=True
        )
        gen.fit(positive_data)
        names = gen.get_feature_names_out(["a", "b", "c"])
        assert len(names) == 9
        assert "a" in names
        assert "sqrt(a)" in names
        assert "square(c)" in names


class TestMathematicalFeaturesGeneratorSchema:

    def test_get_parameter_schema(self):
        schema = MathematicalFeaturesGenerator.get_parameter_schema()
        assert "transformations" in schema
        assert "include_original" in schema
        assert "handle_invalid" in schema


# ---- BinningFeaturesGenerator ----

class TestBinningFeaturesGeneratorInit:

    def test_default_init(self):
        binner = BinningFeaturesGenerator()
        assert binner.n_bins == 5
        assert binner.strategy == "quantile"
        assert binner.encode == "ordinal"
        assert binner.include_original is False

    def test_custom_init(self):
        binner = BinningFeaturesGenerator(
            n_bins=10, strategy="uniform", encode="onehot", include_original=True
        )
        assert binner.n_bins == 10
        assert binner.strategy == "uniform"
        assert binner.encode == "onehot"
        assert binner.include_original is True


class TestBinningFeaturesGeneratorFit:

    def test_fit_stores_bin_edges(self, general_data):
        binner = BinningFeaturesGenerator(n_bins=5, strategy="quantile")
        binner.fit(general_data)
        assert binner.bin_edges_ is not None
        assert len(binner.bin_edges_) == general_data.shape[1]

    def test_uniform_strategy(self, general_data):
        binner = BinningFeaturesGenerator(n_bins=4, strategy="uniform")
        binner.fit(general_data)
        # Each feature should have 5 edges (4 bins + 1)
        for edges in binner.bin_edges_:
            assert len(edges) == 5

    def test_transform_before_fit_raises(self, general_data):
        binner = BinningFeaturesGenerator()
        with pytest.raises(RuntimeError):
            binner.transform(general_data)

    def test_invalid_strategy_raises(self, general_data):
        binner = BinningFeaturesGenerator(strategy="invalid")
        with pytest.raises(ValueError, match="Unknown strategy"):
            binner.fit(general_data)


class TestBinningFeaturesGeneratorTransform:

    def test_ordinal_encoding_shape(self, general_data):
        binner = BinningFeaturesGenerator(n_bins=5, strategy="quantile", encode="ordinal")
        X_binned = binner.fit_transform(general_data)
        assert X_binned.shape == general_data.shape

    def test_ordinal_encoding_values(self, general_data):
        binner = BinningFeaturesGenerator(n_bins=4, strategy="uniform", encode="ordinal")
        X_binned = binner.fit_transform(general_data)
        # Values should be non-negative integers
        unique_vals = np.unique(X_binned)
        assert all(v >= 0 for v in unique_vals)

    def test_onehot_encoding_shape(self, general_data):
        binner = BinningFeaturesGenerator(
            n_bins=4, strategy="uniform", encode="onehot"
        )
        X_binned = binner.fit_transform(general_data)
        # For 4 features with 4 bins each = 16 columns
        assert X_binned.shape[0] == general_data.shape[0]
        assert X_binned.shape[1] >= general_data.shape[1]

    def test_include_original(self, general_data):
        binner = BinningFeaturesGenerator(
            n_bins=4, strategy="quantile", encode="ordinal", include_original=True
        )
        X_binned = binner.fit_transform(general_data)
        # Original (4) + binned (4) = 8
        assert X_binned.shape[1] == 2 * general_data.shape[1]

    def test_kmeans_strategy(self, general_data):
        binner = BinningFeaturesGenerator(n_bins=3, strategy="kmeans")
        X_binned = binner.fit_transform(general_data)
        assert X_binned.shape[0] == general_data.shape[0]


class TestBinningFeaturesGeneratorSchema:

    def test_get_parameter_schema(self):
        schema = BinningFeaturesGenerator.get_parameter_schema()
        assert "n_bins" in schema
        assert "strategy" in schema
        assert "encode" in schema
        assert "include_original" in schema
