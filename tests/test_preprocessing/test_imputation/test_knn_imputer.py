"""Tests for KNNImputer transformer."""

import numpy as np
import pytest

from tuiml.preprocessing.imputation.knn_imputer import KNNImputer


@pytest.fixture
def data_with_nans():
    return np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [np.nan, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
    ])


def test_init_defaults():
    imputer = KNNImputer()
    assert imputer.n_neighbors == 5
    assert imputer.weights == "uniform"
    assert imputer.columns is None


def test_fit_returns_self(data_with_nans):
    imputer = KNNImputer(n_neighbors=2)
    result = imputer.fit(data_with_nans)
    assert result is imputer


def test_imputation_fills_nans(data_with_nans):
    imputer = KNNImputer(n_neighbors=2)
    X_imputed = imputer.fit_transform(data_with_nans)
    assert not np.any(np.isnan(X_imputed))


def test_imputation_uses_neighbors(data_with_nans):
    imputer = KNNImputer(n_neighbors=2)
    X_imputed = imputer.fit_transform(data_with_nans)
    # The imputed value for row 2, col 0 should be reasonable
    # (interpolated from neighbors)
    assert 0.0 < X_imputed[2, 0] < 15.0


def test_distance_weights():
    X = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [np.nan, 10.0],
        [10.0, 10.0],
    ])
    imputer = KNNImputer(n_neighbors=2, weights="distance")
    X_imputed = imputer.fit_transform(X)
    assert not np.any(np.isnan(X_imputed))


def test_invalid_weights_raises():
    with pytest.raises(ValueError):
        KNNImputer(weights="invalid")


def test_transform_before_fit_raises():
    imputer = KNNImputer()
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(RuntimeError):
        imputer.transform(X)


def test_get_parameter_schema():
    schema = KNNImputer.get_parameter_schema()
    assert "n_neighbors" in schema
    assert "weights" in schema
    assert "columns" in schema


def test_no_nans_passthrough():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    imputer = KNNImputer(n_neighbors=2)
    X_imputed = imputer.fit_transform(X)
    np.testing.assert_allclose(X_imputed, X)
