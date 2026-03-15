"""Tests for OneHotEncoder transformer."""

import numpy as np
import pytest

from tuiml.preprocessing.encoding.nominal_to_binary import OneHotEncoder


def test_init_defaults():
    encoder = OneHotEncoder()
    assert encoder.categories is None
    assert encoder.drop is None
    assert encoder.columns is None


def test_fit_returns_self():
    X = np.array([[0], [1], [2]])
    encoder = OneHotEncoder()
    result = encoder.fit(X)
    assert result is encoder


def test_basic_encoding():
    X = np.array([[0], [1], [2], [0]])
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)
    assert X_encoded.shape == (4, 3)
    # Row 0: category 0 -> [1, 0, 0]
    np.testing.assert_allclose(X_encoded[0], [1, 0, 0])
    # Row 1: category 1 -> [0, 1, 0]
    np.testing.assert_allclose(X_encoded[1], [0, 1, 0])
    # Row 2: category 2 -> [0, 0, 1]
    np.testing.assert_allclose(X_encoded[2], [0, 0, 1])


def test_drop_first():
    X = np.array([[0], [1], [2], [0]])
    encoder = OneHotEncoder(drop="first")
    X_encoded = encoder.fit_transform(X)
    assert X_encoded.shape == (4, 2)  # 3 categories minus 1


def test_drop_if_binary():
    X = np.array([[0], [1], [0], [1]])
    encoder = OneHotEncoder(drop="if_binary")
    X_encoded = encoder.fit_transform(X)
    assert X_encoded.shape == (4, 1)  # Binary: drop one


def test_columns_parameter():
    X = np.array([[1.0, 0], [2.0, 1], [3.0, 2]])
    encoder = OneHotEncoder(columns=[1])
    X_encoded = encoder.fit_transform(X)
    # Column 0 passed through, column 1 one-hot encoded to 3 columns
    assert X_encoded.shape == (3, 4)
    np.testing.assert_allclose(X_encoded[:, 0], [1.0, 2.0, 3.0])


def test_transform_before_fit_raises():
    encoder = OneHotEncoder()
    X = np.array([[0], [1]])
    with pytest.raises(RuntimeError):
        encoder.transform(X)


def test_get_parameter_schema():
    schema = OneHotEncoder.get_parameter_schema()
    assert "categories" in schema
    assert "drop" in schema
    assert "columns" in schema


def test_get_feature_names_out():
    X = np.array([[0], [1], [2]])
    encoder = OneHotEncoder()
    encoder.fit(X, feature_names=["color"])
    names = encoder.get_feature_names_out()
    assert len(names) == 3
    assert "color_0.0" in names or "color_0" in names
