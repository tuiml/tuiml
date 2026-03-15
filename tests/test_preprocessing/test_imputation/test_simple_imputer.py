"""Tests for SimpleImputer transformer."""

import numpy as np
import pytest

from tuiml.preprocessing.imputation.simple_imputer import SimpleImputer


@pytest.fixture
def data_with_nans():
    return np.array([
        [1.0, 10.0],
        [2.0, np.nan],
        [np.nan, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
    ])


def test_init_defaults():
    imputer = SimpleImputer()
    assert imputer.strategy == "mean"
    assert imputer.fill_value is None
    assert imputer.columns is None


def test_mean_strategy(data_with_nans):
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(data_with_nans)
    # Column 0: mean of [1, 2, 4, 5] = 3.0
    np.testing.assert_allclose(X_imputed[2, 0], 3.0)
    # Column 1: mean of [10, 30, 40, 50] = 32.5
    np.testing.assert_allclose(X_imputed[1, 1], 32.5)
    # No NaN should remain
    assert not np.any(np.isnan(X_imputed))


def test_median_strategy(data_with_nans):
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(data_with_nans)
    # Column 0: median of [1, 2, 4, 5] = 3.0
    np.testing.assert_allclose(X_imputed[2, 0], 3.0)
    assert not np.any(np.isnan(X_imputed))


def test_most_frequent_strategy():
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


def test_constant_strategy():
    X = np.array([[1.0, np.nan], [np.nan, 3.0]])
    imputer = SimpleImputer(strategy="constant", fill_value=-1.0)
    X_imputed = imputer.fit_transform(X)
    np.testing.assert_allclose(X_imputed[0, 1], -1.0)
    np.testing.assert_allclose(X_imputed[1, 0], -1.0)


def test_columns_parameter():
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


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        SimpleImputer(strategy="invalid")


def test_constant_without_fill_value_raises():
    X = np.array([[1.0, np.nan], [2.0, 3.0]])
    imputer = SimpleImputer(strategy="constant", fill_value=None)
    with pytest.raises(ValueError):
        imputer.fit(X)


def test_transform_before_fit_raises():
    imputer = SimpleImputer()
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(RuntimeError):
        imputer.transform(X)


def test_get_parameter_schema():
    schema = SimpleImputer.get_parameter_schema()
    assert "strategy" in schema
    assert "fill_value" in schema
    assert "columns" in schema


def test_fit_returns_self(data_with_nans):
    imputer = SimpleImputer()
    result = imputer.fit(data_with_nans)
    assert result is imputer
