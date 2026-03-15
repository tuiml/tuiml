"""Tests for DifferenceTransformer."""

import numpy as np
import pytest

from tuiml.preprocessing.timeseries.delta import DifferenceTransformer


def test_init_defaults():
    dt = DifferenceTransformer()
    assert dt.lag == -1
    assert dt.columns is None
    assert dt.fill_with_missing is True
    assert dt.invert_selection is False


def test_fit_returns_self():
    X = np.array([[10], [20], [30]])
    dt = DifferenceTransformer()
    result = dt.fit(X)
    assert result is dt


def test_basic_difference_lag_neg1():
    X = np.array([[10.0], [12.0], [11.0], [15.0]])
    dt = DifferenceTransformer(lag=-1)
    X_diff = dt.fit_transform(X)
    assert X_diff.shape == (4, 1)
    assert np.isnan(X_diff[0, 0])
    np.testing.assert_allclose(X_diff[1, 0], 2.0)
    np.testing.assert_allclose(X_diff[2, 0], -1.0)
    np.testing.assert_allclose(X_diff[3, 0], 4.0)


def test_positive_lag():
    X = np.array([[10.0], [12.0], [11.0], [15.0]])
    dt = DifferenceTransformer(lag=1)
    X_diff = dt.fit_transform(X)
    assert X_diff.shape == (4, 1)
    # lag=1 means current - future
    np.testing.assert_allclose(X_diff[0, 0], -2.0)
    np.testing.assert_allclose(X_diff[1, 0], 1.0)
    np.testing.assert_allclose(X_diff[2, 0], -4.0)
    assert np.isnan(X_diff[3, 0])


def test_lag_neg2():
    X = np.array([[10.0], [20.0], [30.0], [40.0]])
    dt = DifferenceTransformer(lag=-2)
    X_diff = dt.fit_transform(X)
    assert np.isnan(X_diff[0, 0])
    assert np.isnan(X_diff[1, 0])
    np.testing.assert_allclose(X_diff[2, 0], 20.0)  # 30 - 10
    np.testing.assert_allclose(X_diff[3, 0], 20.0)  # 40 - 20


def test_fill_with_missing_false():
    X = np.array([[10.0], [20.0], [30.0], [40.0]])
    dt = DifferenceTransformer(lag=-1, fill_with_missing=False)
    X_diff = dt.fit_transform(X)
    # First row should be removed (boundary)
    assert X_diff.shape[0] == 3
    np.testing.assert_allclose(X_diff[0, 0], 10.0)


def test_columns_parameter():
    X = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    dt = DifferenceTransformer(lag=-1, columns=[0])
    X_diff = dt.fit_transform(X)
    assert X_diff.shape == (3, 2)
    # Column 0 should be differenced
    assert np.isnan(X_diff[0, 0])
    np.testing.assert_allclose(X_diff[1, 0], 10.0)
    # Column 1 should be unchanged
    np.testing.assert_allclose(X_diff[:, 1], [100.0, 200.0, 300.0])


def test_invert_selection():
    X = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    dt = DifferenceTransformer(lag=-1, columns=[0], invert_selection=True)
    X_diff = dt.fit_transform(X)
    # Column 0 should be unchanged (inverted means apply to non-specified)
    np.testing.assert_allclose(X_diff[:, 0], [10.0, 20.0, 30.0])
    # Column 1 should be differenced
    assert np.isnan(X_diff[0, 1])
    np.testing.assert_allclose(X_diff[1, 1], 100.0)


def test_zero_lag():
    X = np.array([[10.0], [20.0], [30.0]])
    dt = DifferenceTransformer(lag=0)
    X_diff = dt.fit_transform(X)
    # lag=0 means current - current = 0
    np.testing.assert_allclose(X_diff[:, 0], [0.0, 0.0, 0.0])


def test_transform_before_fit_raises():
    dt = DifferenceTransformer()
    X = np.array([[1], [2]])
    with pytest.raises(RuntimeError):
        dt.transform(X)


def test_feature_names_out():
    X = np.array([[10.0], [20.0], [30.0]])
    dt = DifferenceTransformer(lag=-1)
    dt.fit(X)
    names = dt.get_feature_names_out()
    assert len(names) == 1
    assert "d-1" in names[0]


def test_feature_count_mismatch_raises():
    X_train = np.array([[10.0, 20.0], [30.0, 40.0]])
    X_test = np.array([[10.0], [20.0]])
    dt = DifferenceTransformer()
    dt.fit(X_train)
    with pytest.raises(ValueError):
        dt.transform(X_test)


def test_get_parameter_schema():
    schema = DifferenceTransformer.get_parameter_schema()
    assert "lag" in schema
    assert "columns" in schema
    assert "fill_with_missing" in schema
    assert "invert_selection" in schema
