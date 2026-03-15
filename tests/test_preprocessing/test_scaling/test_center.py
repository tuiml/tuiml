"""Tests for CenterScaler transformer."""

import numpy as np
import pytest

from tuiml.preprocessing.scaling.center import CenterScaler


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.randn(100, 3) * 10 + 5


def test_init_defaults():
    scaler = CenterScaler()
    assert scaler.columns is None


def test_fit_returns_self(sample_data):
    scaler = CenterScaler()
    result = scaler.fit(sample_data)
    assert result is scaler


def test_fit_transform_zero_mean(sample_data):
    scaler = CenterScaler()
    X_centered = scaler.fit_transform(sample_data)
    np.testing.assert_allclose(X_centered.mean(axis=0), 0.0, atol=1e-10)


def test_std_unchanged(sample_data):
    scaler = CenterScaler()
    original_std = sample_data.std(axis=0)
    X_centered = scaler.fit_transform(sample_data)
    np.testing.assert_allclose(X_centered.std(axis=0), original_std, atol=1e-10)


def test_inverse_transform_recovers_original(sample_data):
    scaler = CenterScaler()
    X_centered = scaler.fit_transform(sample_data)
    X_recovered = scaler.inverse_transform(X_centered)
    np.testing.assert_allclose(X_recovered, sample_data, atol=1e-10)


def test_columns_parameter():
    X = np.array([[1.0, 100.0, 10.0],
                  [2.0, 200.0, 20.0],
                  [3.0, 300.0, 30.0]])
    scaler = CenterScaler(columns=[1])
    X_centered = scaler.fit_transform(X)
    # Column 0 and 2 should be unchanged
    np.testing.assert_allclose(X_centered[:, 0], X[:, 0])
    np.testing.assert_allclose(X_centered[:, 2], X[:, 2])
    # Column 1 should be centered
    np.testing.assert_allclose(X_centered[:, 1].mean(), 0.0, atol=1e-10)


def test_transform_before_fit_raises():
    scaler = CenterScaler()
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(RuntimeError):
        scaler.transform(X)


def test_get_parameter_schema():
    schema = CenterScaler.get_parameter_schema()
    assert "columns" in schema


def test_feature_count_mismatch_raises():
    scaler = CenterScaler()
    X_train = np.array([[1, 2], [3, 4]])
    scaler.fit(X_train)
    X_test = np.array([[1, 2, 3]])
    with pytest.raises(ValueError):
        scaler.transform(X_test)
