"""Tests for MDLDiscretizer transformer."""

import numpy as np
import pytest

from tuiml.preprocessing.discretization.mdl import MDLDiscretizer


def test_init_defaults():
    disc = MDLDiscretizer()
    assert disc.min_instances == 10
    assert disc.columns is None


def test_fit_returns_self():
    X = np.arange(100).reshape(-1, 1).astype(float)
    y = np.array([0] * 50 + [1] * 50)
    disc = MDLDiscretizer(min_instances=5)
    result = disc.fit(X, y)
    assert result is disc


def test_supervised_discretization():
    np.random.seed(42)
    X = np.arange(100).reshape(-1, 1).astype(float)
    y = np.array([0] * 50 + [1] * 50)
    disc = MDLDiscretizer(min_instances=5)
    X_binned = disc.fit_transform(X, y)
    # Should produce at least 2 distinct bins for this clear separation
    unique = np.unique(X_binned)
    assert len(unique) >= 2


def test_cut_points_property():
    X = np.arange(100).reshape(-1, 1).astype(float)
    y = np.array([0] * 50 + [1] * 50)
    disc = MDLDiscretizer(min_instances=5)
    disc.fit(X, y)
    cuts = disc.cut_points_
    assert isinstance(cuts, dict)


def test_no_clear_separation():
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = np.random.randint(0, 2, 100)
    disc = MDLDiscretizer(min_instances=20)
    X_binned = disc.fit_transform(X, y)
    # May not find any cut points with random data
    assert X_binned.shape == X.shape


def test_transform_before_fit_raises():
    disc = MDLDiscretizer()
    X = np.array([[1], [2]])
    with pytest.raises(RuntimeError):
        disc.transform(X)


def test_get_parameter_schema():
    schema = MDLDiscretizer.get_parameter_schema()
    assert "min_instances" in schema
    assert "columns" in schema
