"""Tests for ValueClipper transformer."""

import numpy as np
import pytest

from tuiml.preprocessing.outliers.clip import ValueClipper


def test_init_defaults():
    clipper = ValueClipper()
    assert clipper.lower is None
    assert clipper.upper is None
    assert clipper.percentile is None
    assert clipper.columns is None


def test_fit_returns_self():
    X = np.array([[1], [5], [10]])
    clipper = ValueClipper(lower=0, upper=10)
    result = clipper.fit(X)
    assert result is clipper


def test_fixed_bounds():
    X = np.array([[-10], [5], [20]])
    clipper = ValueClipper(lower=0, upper=10)
    X_clipped = clipper.fit_transform(X)
    np.testing.assert_allclose(X_clipped.flatten(), [0.0, 5.0, 10.0])


def test_lower_only():
    X = np.array([[-10], [5], [20]])
    clipper = ValueClipper(lower=0)
    X_clipped = clipper.fit_transform(X)
    np.testing.assert_allclose(X_clipped.flatten(), [0.0, 5.0, 20.0])


def test_upper_only():
    X = np.array([[-10], [5], [20]])
    clipper = ValueClipper(upper=10)
    X_clipped = clipper.fit_transform(X)
    np.testing.assert_allclose(X_clipped.flatten(), [-10.0, 5.0, 10.0])


def test_percentile_bounds():
    np.random.seed(42)
    X = np.random.randn(1000, 1)
    clipper = ValueClipper(percentile=(1, 99))
    X_clipped = clipper.fit_transform(X)
    assert X_clipped.min() >= np.percentile(X, 1) - 1e-10
    assert X_clipped.max() <= np.percentile(X, 99) + 1e-10


def test_columns_parameter():
    X = np.array([[-10.0, -10.0], [5.0, 5.0], [20.0, 20.0]])
    clipper = ValueClipper(lower=0, upper=10, columns=[0])
    X_clipped = clipper.fit_transform(X)
    np.testing.assert_allclose(X_clipped[:, 0], [0.0, 5.0, 10.0])
    # Column 1 should be unchanged
    np.testing.assert_allclose(X_clipped[:, 1], [-10.0, 5.0, 20.0])


def test_transform_before_fit_raises():
    clipper = ValueClipper(lower=0, upper=10)
    X = np.array([[1], [2]])
    with pytest.raises(RuntimeError):
        clipper.transform(X)


def test_get_parameter_schema():
    schema = ValueClipper.get_parameter_schema()
    assert "lower" in schema
    assert "upper" in schema
    assert "percentile" in schema
    assert "columns" in schema
