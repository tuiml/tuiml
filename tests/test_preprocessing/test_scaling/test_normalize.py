"""Tests for MinMaxScaler transformer."""

import numpy as np
import pytest

from tuiml.preprocessing.scaling.normalize import MinMaxScaler


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.randn(100, 3) * 10 + 5


def test_init_defaults():
    scaler = MinMaxScaler()
    assert scaler.scale == 1.0
    assert scaler.translation == 0.0
    assert scaler.columns is None


def test_fit_returns_self(sample_data):
    scaler = MinMaxScaler()
    result = scaler.fit(sample_data)
    assert result is scaler


def test_fit_transform_range_01(sample_data):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(sample_data)
    np.testing.assert_allclose(X_scaled.min(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(X_scaled.max(axis=0), 1.0, atol=1e-10)


def test_custom_range():
    np.random.seed(42)
    X = np.random.randn(50, 2)
    scaler = MinMaxScaler(scale=2.0, translation=-1.0)
    X_scaled = scaler.fit_transform(X)
    np.testing.assert_allclose(X_scaled.min(axis=0), -1.0, atol=1e-10)
    np.testing.assert_allclose(X_scaled.max(axis=0), 1.0, atol=1e-10)


def test_inverse_transform_recovers_original(sample_data):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(sample_data)
    X_recovered = scaler.inverse_transform(X_scaled)
    np.testing.assert_allclose(X_recovered, sample_data, atol=1e-10)


def test_columns_parameter():
    X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    scaler = MinMaxScaler(columns=[0])
    X_scaled = scaler.fit_transform(X)
    # Column 0 should be in [0, 1]
    np.testing.assert_allclose(X_scaled[0, 0], 0.0, atol=1e-10)
    np.testing.assert_allclose(X_scaled[2, 0], 1.0, atol=1e-10)
    # Column 1 should be unchanged
    np.testing.assert_allclose(X_scaled[:, 1], X[:, 1])


def test_transform_before_fit_raises():
    scaler = MinMaxScaler()
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(RuntimeError):
        scaler.transform(X)


def test_get_parameter_schema():
    schema = MinMaxScaler.get_parameter_schema()
    assert "scale" in schema
    assert "translation" in schema
    assert "columns" in schema


def test_zero_range_column():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    assert not np.any(np.isnan(X_scaled))
    assert not np.any(np.isinf(X_scaled))


def test_feature_count_mismatch_raises():
    scaler = MinMaxScaler()
    X_train = np.array([[1, 2], [3, 4]])
    scaler.fit(X_train)
    X_test = np.array([[1, 2, 3]])
    with pytest.raises(ValueError):
        scaler.transform(X_test)
