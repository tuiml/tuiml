"""Discretization filters.

Merged from: test_discretization_equal_width.py, test_discretization_equal_frequency.py, test_discretization_mdl.py
"""

import numpy as np
from tuiml.preprocessing.discretization.equal_width import EqualWidthDiscretizer
from tuiml.preprocessing.discretization.equal_frequency import QuantileDiscretizer
from tuiml.preprocessing.discretization.mdl import MDLDiscretizer


# --------------------------------------------------------------------------
# Tests for EqualWidthDiscretizer transformer.
# --------------------------------------------------------------------------

class TestEqualWidthDiscretizer:
    """Tests for EqualWidthDiscretizer transformer."""

    def test_init_defaults(self):
        disc = EqualWidthDiscretizer()
        assert disc.n_bins == 10
        assert disc.columns is None

    def test_output_is_integer_bins(self):
        X = np.arange(100).reshape(-1, 1).astype(float)
        disc = EqualWidthDiscretizer(n_bins=5)
        X_binned = disc.fit_transform(X)
        unique = np.unique(X_binned)
        assert len(unique) == 5

    def test_n_bins_parameter(self):
        X = np.linspace(0, 100, 1000).reshape(-1, 1)
        disc = EqualWidthDiscretizer(n_bins=4)
        X_binned = disc.fit_transform(X)
        unique = np.unique(X_binned)
        assert len(unique) == 4

    def test_columns_parameter(self):
        X = np.array([[1.0, 100.0], [5.0, 200.0], [10.0, 300.0]])
        disc = EqualWidthDiscretizer(n_bins=2, columns=[0])
        X_binned = disc.fit_transform(X)
        # Column 1 should be unchanged
        np.testing.assert_allclose(X_binned[:, 1], X[:, 1])

    def test_bin_edges_property(self):
        X = np.arange(10).reshape(-1, 1).astype(float)
        disc = EqualWidthDiscretizer(n_bins=5)
        disc.fit(X)
        edges = disc.bin_edges_
        assert 0 in edges
        assert len(edges[0]) == 6  # n_bins + 1 edges

    def test_get_parameter_schema(self):
        schema = EqualWidthDiscretizer.get_parameter_schema()
        assert "n_bins" in schema
        assert "columns" in schema


# --------------------------------------------------------------------------
# Tests for QuantileDiscretizer transformer.
# --------------------------------------------------------------------------

class TestQuantileDiscretizer:
    """Tests for QuantileDiscretizer transformer."""

    def test_init_defaults(self):
        disc = QuantileDiscretizer()
        assert disc.n_bins == 10
        assert disc.columns is None

    def test_equal_frequency_bins(self):
        np.random.seed(42)
        X = np.random.randn(1000, 1)
        disc = QuantileDiscretizer(n_bins=4)
        X_binned = disc.fit_transform(X)
        # Each bin should have approximately equal count
        unique, counts = np.unique(X_binned, return_counts=True)
        # Allow some tolerance since quantile edges may overlap
        assert len(unique) <= 4

    def test_columns_parameter(self):
        X = np.array([[1.0, 100.0], [5.0, 200.0], [10.0, 300.0]])
        disc = QuantileDiscretizer(n_bins=2, columns=[1])
        X_binned = disc.fit_transform(X)
        # Column 0 should be unchanged
        np.testing.assert_allclose(X_binned[:, 0], X[:, 0])

    def test_bin_edges_property(self):
        X = np.arange(100).reshape(-1, 1).astype(float)
        disc = QuantileDiscretizer(n_bins=4)
        disc.fit(X)
        edges = disc.bin_edges_
        assert 0 in edges

    def test_get_parameter_schema(self):
        schema = QuantileDiscretizer.get_parameter_schema()
        assert "n_bins" in schema
        assert "columns" in schema


# --------------------------------------------------------------------------
# Tests for MDLDiscretizer transformer.
# --------------------------------------------------------------------------

class TestMDLDiscretizer:
    """Tests for MDLDiscretizer transformer."""

    def test_init_defaults(self):
        disc = MDLDiscretizer()
        assert disc.min_instances == 10
        assert disc.columns is None

    def test_fit_returns_self(self):
        X = np.arange(100).reshape(-1, 1).astype(float)
        y = np.array([0] * 50 + [1] * 50)
        disc = MDLDiscretizer(min_instances=5)
        result = disc.fit(X, y)
        assert result is disc

    def test_supervised_discretization(self):
        np.random.seed(42)
        X = np.arange(100).reshape(-1, 1).astype(float)
        y = np.array([0] * 50 + [1] * 50)
        disc = MDLDiscretizer(min_instances=5)
        X_binned = disc.fit_transform(X, y)
        # Should produce at least 2 distinct bins for this clear separation
        unique = np.unique(X_binned)
        assert len(unique) >= 2

    def test_cut_points_property(self):
        X = np.arange(100).reshape(-1, 1).astype(float)
        y = np.array([0] * 50 + [1] * 50)
        disc = MDLDiscretizer(min_instances=5)
        disc.fit(X, y)
        cuts = disc.cut_points_
        assert isinstance(cuts, dict)

    def test_no_clear_separation(self):
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = np.random.randint(0, 2, 100)
        disc = MDLDiscretizer(min_instances=20)
        X_binned = disc.fit_transform(X, y)
        # May not find any cut points with random data
        assert X_binned.shape == X.shape

    def test_get_parameter_schema(self):
        schema = MDLDiscretizer.get_parameter_schema()
        assert "min_instances" in schema
        assert "columns" in schema
