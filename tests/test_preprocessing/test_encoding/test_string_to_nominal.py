"""Tests for LabelEncoder transformer."""

import numpy as np
import pytest

from tuiml.preprocessing.encoding.string_to_nominal import LabelEncoder


def test_init_defaults():
    encoder = LabelEncoder()
    assert encoder.columns is None


def test_fit_returns_self():
    X = np.array([["cat"], ["dog"]], dtype=object)
    encoder = LabelEncoder()
    result = encoder.fit(X)
    assert result is encoder


def test_basic_encoding():
    X = np.array([["cat"], ["dog"], ["cat"]], dtype=object)
    encoder = LabelEncoder()
    X_encoded = encoder.fit_transform(X)
    # cat=0, dog=1 (order of first occurrence)
    assert X_encoded[0, 0] == 0.0
    assert X_encoded[1, 0] == 1.0
    assert X_encoded[2, 0] == 0.0


def test_auto_detect_string_columns():
    X = np.array([[1.0, "cat"], [2.0, "dog"]], dtype=object)
    encoder = LabelEncoder()
    X_encoded = encoder.fit_transform(X)
    # Column 0 should stay numeric, column 1 should be encoded
    np.testing.assert_allclose(X_encoded[:, 0], [1.0, 2.0])
    assert X_encoded[0, 1] == 0.0
    assert X_encoded[1, 1] == 1.0


def test_unseen_category_maps_to_negative():
    X_train = np.array([["cat"], ["dog"]], dtype=object)
    X_test = np.array([["cat"], ["bird"]], dtype=object)
    encoder = LabelEncoder()
    encoder.fit(X_train)
    X_encoded = encoder.transform(X_test)
    assert X_encoded[0, 0] == 0.0
    assert X_encoded[1, 0] == -1.0


def test_categories_property():
    X = np.array([["a"], ["b"], ["c"]], dtype=object)
    encoder = LabelEncoder()
    encoder.fit(X)
    cats = encoder.categories_
    assert 0 in cats
    assert len(cats[0]) == 3


def test_transform_before_fit_raises():
    encoder = LabelEncoder()
    X = np.array([["a"]], dtype=object)
    with pytest.raises(RuntimeError):
        encoder.transform(X)


def test_get_parameter_schema():
    schema = LabelEncoder.get_parameter_schema()
    assert "columns" in schema
