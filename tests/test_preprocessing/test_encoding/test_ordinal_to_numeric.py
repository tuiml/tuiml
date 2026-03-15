"""Tests for OrdinalEncoder transformer."""

import numpy as np
import pytest

from tuiml.preprocessing.encoding.ordinal_to_numeric import OrdinalEncoder


def test_init_defaults():
    encoder = OrdinalEncoder()
    assert encoder.categories is None
    assert encoder.columns is None


def test_fit_returns_self():
    X = np.array([["low"], ["medium"], ["high"]], dtype=object)
    encoder = OrdinalEncoder(categories=[["low", "medium", "high"]])
    result = encoder.fit(X)
    assert result is encoder


def test_explicit_order():
    X = np.array([["low"], ["medium"], ["high"]], dtype=object)
    encoder = OrdinalEncoder(categories=[["low", "medium", "high"]])
    X_encoded = encoder.fit_transform(X)
    np.testing.assert_allclose(X_encoded.flatten(), [0.0, 1.0, 2.0])


def test_inferred_order():
    X = np.array([["cat"], ["dog"], ["cat"], ["bird"]], dtype=object)
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)
    # Order inferred from first occurrence: cat=0, dog=1, bird=2
    assert X_encoded[0, 0] == 0.0
    assert X_encoded[1, 0] == 1.0
    assert X_encoded[3, 0] == 2.0


def test_transform_before_fit_raises():
    encoder = OrdinalEncoder()
    X = np.array([["a"], ["b"]], dtype=object)
    with pytest.raises(RuntimeError):
        encoder.transform(X)


def test_get_parameter_schema():
    schema = OrdinalEncoder.get_parameter_schema()
    assert "categories" in schema
    assert "columns" in schema


def test_unknown_category_maps_to_negative():
    X_train = np.array([["a"], ["b"]], dtype=object)
    X_test = np.array([["a"], ["c"]], dtype=object)
    encoder = OrdinalEncoder()
    encoder.fit(X_train)
    X_encoded = encoder.transform(X_test)
    assert X_encoded[0, 0] == 0.0
    assert X_encoded[1, 0] == -1.0


def test_multiple_columns():
    X = np.array([["low", "red"], ["high", "blue"]], dtype=object)
    encoder = OrdinalEncoder(
        categories=[["low", "high"], ["red", "blue"]]
    )
    X_encoded = encoder.fit_transform(X)
    assert X_encoded.shape == (2, 2)
    np.testing.assert_allclose(X_encoded[0], [0.0, 0.0])
    np.testing.assert_allclose(X_encoded[1], [1.0, 1.0])
