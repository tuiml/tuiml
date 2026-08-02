"""Tests for EqualWidthDiscretizer transformer."""

import numpy as np

from tuiml.preprocessing.discretization.equal_width import EqualWidthDiscretizer


def test_init_defaults():
    disc = EqualWidthDiscretizer()
    assert disc.n_bins == 10
    assert disc.columns is None


def test_output_is_integer_bins():
    X = np.arange(100).reshape(-1, 1).astype(float)
    disc = EqualWidthDiscretizer(n_bins=5)
    X_binned = disc.fit_transform(X)
    unique = np.unique(X_binned)
    assert len(unique) == 5


def test_n_bins_parameter():
    X = np.linspace(0, 100, 1000).reshape(-1, 1)
    disc = EqualWidthDiscretizer(n_bins=4)
    X_binned = disc.fit_transform(X)
    unique = np.unique(X_binned)
    assert len(unique) == 4


def test_columns_parameter():
    X = np.array([[1.0, 100.0], [5.0, 200.0], [10.0, 300.0]])
    disc = EqualWidthDiscretizer(n_bins=2, columns=[0])
    X_binned = disc.fit_transform(X)
    # Column 1 should be unchanged
    np.testing.assert_allclose(X_binned[:, 1], X[:, 1])


def test_bin_edges_property():
    X = np.arange(10).reshape(-1, 1).astype(float)
    disc = EqualWidthDiscretizer(n_bins=5)
    disc.fit(X)
    edges = disc.bin_edges_
    assert 0 in edges
    assert len(edges[0]) == 6  # n_bins + 1 edges


def test_get_parameter_schema():
    schema = EqualWidthDiscretizer.get_parameter_schema()
    assert "n_bins" in schema
    assert "columns" in schema
