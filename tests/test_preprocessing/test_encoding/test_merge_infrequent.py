"""Tests for RareCategoryEncoder transformer."""

import numpy as np
import pytest

from tuiml.preprocessing.encoding.merge_infrequent import RareCategoryEncoder


def test_init_defaults():
    encoder = RareCategoryEncoder()
    assert encoder.min_frequency == 5
    assert encoder.merged_value == -1
    assert encoder.columns is None


def test_fit_returns_self():
    X = np.array([[0], [0], [0], [1], [2]])
    encoder = RareCategoryEncoder(min_frequency=2)
    result = encoder.fit(X)
    assert result is encoder


def test_merge_rare_values():
    # 0 appears 5 times, 1 appears 3 times, 2 appears 1 time
    X = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [2]])
    encoder = RareCategoryEncoder(min_frequency=3, merged_value=-1)
    X_merged = encoder.fit_transform(X)
    # Value 2 (count 1) should be merged to -1
    np.testing.assert_allclose(X_merged[8, 0], -1.0)
    # Value 0 (count 5) should remain
    np.testing.assert_allclose(X_merged[0, 0], 0.0)
    # Value 1 (count 3) should remain
    np.testing.assert_allclose(X_merged[5, 0], 1.0)


def test_proportion_threshold():
    X = np.arange(10).reshape(-1, 1).astype(float)
    # Each value appears once out of 10 = 0.1
    encoder = RareCategoryEncoder(min_frequency=0.2)
    X_merged = encoder.fit_transform(X)
    # All values appear less than 20%, so all should be merged
    for val in X_merged.flatten():
        assert val == -1.0


def test_columns_parameter():
    X = np.array([[0, 0], [0, 1], [0, 1], [1, 1]])
    encoder = RareCategoryEncoder(min_frequency=2, columns=[0])
    X_merged = encoder.fit_transform(X)
    # Column 0: 0 appears 3 times, 1 appears 1 time -> 1 merged
    np.testing.assert_allclose(X_merged[3, 0], -1.0)
    # Column 1 should be unchanged
    np.testing.assert_allclose(X_merged[:, 1], X[:, 1])


def test_transform_before_fit_raises():
    encoder = RareCategoryEncoder()
    X = np.array([[0], [1]])
    with pytest.raises(RuntimeError):
        encoder.transform(X)


def test_get_parameter_schema():
    schema = RareCategoryEncoder.get_parameter_schema()
    assert "min_frequency" in schema
    assert "merged_value" in schema
    assert "columns" in schema


def test_value_maps_property():
    X = np.array([[0], [0], [0], [1]])
    encoder = RareCategoryEncoder(min_frequency=2)
    encoder.fit(X)
    maps = encoder.value_maps_
    assert isinstance(maps, dict)
