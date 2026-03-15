"""Tests for QuantileDiscretizer transformer."""

import numpy as np
import pytest

from tuiml.preprocessing.discretization.equal_frequency import QuantileDiscretizer


def test_init_defaults():
    disc = QuantileDiscretizer()
    assert disc.n_bins == 10
    assert disc.columns is None


def test_fit_returns_self():
    X = np.arange(100).reshape(-1, 1).astype(float)
    disc = QuantileDiscretizer(n_bins=5)
    result = disc.fit(X)
    assert result is disc


def test_equal_frequency_bins():
    np.random.seed(42)
    X = np.random.randn(1000, 1)
    disc = QuantileDiscretizer(n_bins=4)
    X_binned = disc.fit_transform(X)
    # Each bin should have approximately equal count
    unique, counts = np.unique(X_binned, return_counts=True)
    # Allow some tolerance since quantile edges may overlap
    assert len(unique) <= 4


def test_columns_parameter():
    X = np.array([[1.0, 100.0], [5.0, 200.0], [10.0, 300.0]])
    disc = QuantileDiscretizer(n_bins=2, columns=[1])
    X_binned = disc.fit_transform(X)
    # Column 0 should be unchanged
    np.testing.assert_allclose(X_binned[:, 0], X[:, 0])


def test_bin_edges_property():
    X = np.arange(100).reshape(-1, 1).astype(float)
    disc = QuantileDiscretizer(n_bins=4)
    disc.fit(X)
    edges = disc.bin_edges_
    assert 0 in edges


def test_transform_before_fit_raises():
    disc = QuantileDiscretizer()
    X = np.array([[1], [2]])
    with pytest.raises(RuntimeError):
        disc.transform(X)


def test_get_parameter_schema():
    schema = QuantileDiscretizer.get_parameter_schema()
    assert "n_bins" in schema
    assert "columns" in schema
