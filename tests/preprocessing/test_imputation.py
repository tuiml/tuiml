"""Missing-value imputers.

Merged from: test_imputation_simple_imputer.py, test_imputation_knn_imputer.py
"""

import numpy as np
import pytest
from tuiml.preprocessing.imputation.simple_imputer import SimpleImputer
from tuiml.preprocessing.imputation.knn_imputer import KNNImputer


# --------------------------------------------------------------------------
# Tests for SimpleImputer transformer.
# --------------------------------------------------------------------------

class TestSimpleImputer:
    """Tests for SimpleImputer transformer."""

    @pytest.fixture
    def data_with_nans(self):
        return np.array([
            [1.0, 10.0],
            [2.0, np.nan],
            [np.nan, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ])

    def test_init_defaults(self):
        imputer = SimpleImputer()
        assert imputer.strategy == "mean"
        assert imputer.fill_value is None
        assert imputer.columns is None

    def test_mean_strategy(self, data_with_nans):
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(data_with_nans)
        # Column 0: mean of [1, 2, 4, 5] = 3.0
        np.testing.assert_allclose(X_imputed[2, 0], 3.0)
        # Column 1: mean of [10, 30, 40, 50] = 32.5
        np.testing.assert_allclose(X_imputed[1, 1], 32.5)
        # No NaN should remain
        assert not np.any(np.isnan(X_imputed))

    def test_median_strategy(self, data_with_nans):
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(data_with_nans)
        # Column 0: median of [1, 2, 4, 5] = 3.0
        np.testing.assert_allclose(X_imputed[2, 0], 3.0)
        assert not np.any(np.isnan(X_imputed))

    def test_most_frequent_strategy(self):
        X = np.array([
            [1.0, 10.0],
            [2.0, np.nan],
            [1.0, 30.0],
            [1.0, 10.0],
            [np.nan, 10.0],
        ])
        imputer = SimpleImputer(strategy="most_frequent")
        X_imputed = imputer.fit_transform(X)
        # Column 0: mode is 1.0
        np.testing.assert_allclose(X_imputed[4, 0], 1.0)
        # Column 1: mode is 10.0
        np.testing.assert_allclose(X_imputed[1, 1], 10.0)

    def test_constant_strategy(self):
        X = np.array([[1.0, np.nan], [np.nan, 3.0]])
        imputer = SimpleImputer(strategy="constant", fill_value=-1.0)
        X_imputed = imputer.fit_transform(X)
        np.testing.assert_allclose(X_imputed[0, 1], -1.0)
        np.testing.assert_allclose(X_imputed[1, 0], -1.0)

    def test_columns_parameter(self):
        X = np.array([
            [1.0, np.nan, 100.0],
            [np.nan, 20.0, np.nan],
            [3.0, 30.0, 300.0],
        ])
        imputer = SimpleImputer(strategy="mean", columns=[0])
        X_imputed = imputer.fit_transform(X)
        # Column 0 should be imputed
        np.testing.assert_allclose(X_imputed[1, 0], 2.0)
        # Column 1 NaN should remain (not in columns list)
        assert np.isnan(X_imputed[0, 1])

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            SimpleImputer(strategy="invalid")

    def test_constant_without_fill_value_raises(self):
        X = np.array([[1.0, np.nan], [2.0, 3.0]])
        imputer = SimpleImputer(strategy="constant", fill_value=None)
        with pytest.raises(ValueError):
            imputer.fit(X)

    def test_get_parameter_schema(self):
        schema = SimpleImputer.get_parameter_schema()
        assert "strategy" in schema
        assert "fill_value" in schema
        assert "columns" in schema


# --------------------------------------------------------------------------
# Tests for KNNImputer transformer.
# --------------------------------------------------------------------------

class TestKNNImputer:
    """Tests for KNNImputer transformer."""

    @pytest.fixture
    def data_with_nans(self):
        return np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [np.nan, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ])

    def test_init_defaults(self):
        imputer = KNNImputer()
        assert imputer.n_neighbors == 5
        assert imputer.weights == "uniform"
        assert imputer.columns is None

    def test_imputation_fills_nans(self, data_with_nans):
        imputer = KNNImputer(n_neighbors=2)
        X_imputed = imputer.fit_transform(data_with_nans)
        assert not np.any(np.isnan(X_imputed))

    def test_imputation_uses_neighbors(self, data_with_nans):
        imputer = KNNImputer(n_neighbors=2)
        X_imputed = imputer.fit_transform(data_with_nans)
        # The imputed value for row 2, col 0 should be reasonable
        # (interpolated from neighbors)
        assert 0.0 < X_imputed[2, 0] < 15.0

    def test_distance_weights(self):
        X = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [np.nan, 10.0],
            [10.0, 10.0],
        ])
        imputer = KNNImputer(n_neighbors=2, weights="distance")
        X_imputed = imputer.fit_transform(X)
        assert not np.any(np.isnan(X_imputed))

    def test_invalid_weights_raises(self):
        with pytest.raises(ValueError):
            KNNImputer(weights="invalid")

    def test_get_parameter_schema(self):
        schema = KNNImputer.get_parameter_schema()
        assert "n_neighbors" in schema
        assert "weights" in schema
        assert "columns" in schema

    def test_no_nans_passthrough(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        imputer = KNNImputer(n_neighbors=2)
        X_imputed = imputer.fit_transform(X)
        np.testing.assert_allclose(X_imputed, X)
