"""Time-series preprocessing filters.

Merged from: test_timeseries_lag.py, test_timeseries_delta.py
"""

import numpy as np
import pytest
from tuiml.preprocessing.timeseries.lag import LagTransformer
from tuiml.preprocessing.timeseries.delta import DifferenceTransformer


# --------------------------------------------------------------------------
# Tests for LagTransformer.
# --------------------------------------------------------------------------

class TestLagTransformer:
    """Tests for LagTransformer."""

    def test_init_defaults(self):
        lt = LagTransformer()
        assert lt.lag == -1
        assert lt.columns is None
        assert lt.fill_with_missing is True
        assert lt.invert_selection is False

    def test_basic_lag_neg1(self):
        X = np.array([[10.0], [20.0], [30.0], [40.0]])
        lt = LagTransformer(lag=-1)
        X_lag = lt.fit_transform(X)
        assert X_lag.shape == (4, 1)
        assert np.isnan(X_lag[0, 0])
        np.testing.assert_allclose(X_lag[1, 0], 10.0)
        np.testing.assert_allclose(X_lag[2, 0], 20.0)
        np.testing.assert_allclose(X_lag[3, 0], 30.0)

    def test_positive_lag(self):
        X = np.array([[10.0], [20.0], [30.0], [40.0]])
        lt = LagTransformer(lag=1)
        X_lag = lt.fit_transform(X)
        assert X_lag.shape == (4, 1)
        np.testing.assert_allclose(X_lag[0, 0], 20.0)
        np.testing.assert_allclose(X_lag[1, 0], 30.0)
        np.testing.assert_allclose(X_lag[2, 0], 40.0)
        assert np.isnan(X_lag[3, 0])

    def test_lag_neg2(self):
        X = np.array([[10.0], [20.0], [30.0], [40.0]])
        lt = LagTransformer(lag=-2)
        X_lag = lt.fit_transform(X)
        assert np.isnan(X_lag[0, 0])
        assert np.isnan(X_lag[1, 0])
        np.testing.assert_allclose(X_lag[2, 0], 10.0)
        np.testing.assert_allclose(X_lag[3, 0], 20.0)

    def test_fill_with_missing_false(self):
        X = np.array([[10.0], [20.0], [30.0], [40.0]])
        lt = LagTransformer(lag=-1, fill_with_missing=False)
        X_lag = lt.fit_transform(X)
        # First row should be removed (boundary)
        assert X_lag.shape[0] == 3
        np.testing.assert_allclose(X_lag[0, 0], 10.0)
        np.testing.assert_allclose(X_lag[1, 0], 20.0)
        np.testing.assert_allclose(X_lag[2, 0], 30.0)

    def test_columns_parameter(self):
        X = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
        lt = LagTransformer(lag=-1, columns=[0])
        X_lag = lt.fit_transform(X)
        assert X_lag.shape == (3, 2)
        # Column 0 should be lagged
        assert np.isnan(X_lag[0, 0])
        np.testing.assert_allclose(X_lag[1, 0], 10.0)
        # Column 1 should be unchanged
        np.testing.assert_allclose(X_lag[:, 1], [100.0, 200.0, 300.0])

    def test_invert_selection(self):
        X = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
        lt = LagTransformer(lag=-1, columns=[0], invert_selection=True)
        X_lag = lt.fit_transform(X)
        # Column 0 should be unchanged (inverted means apply to non-specified)
        np.testing.assert_allclose(X_lag[:, 0], [10.0, 20.0, 30.0])
        # Column 1 should be lagged
        assert np.isnan(X_lag[0, 1])
        np.testing.assert_allclose(X_lag[1, 1], 100.0)

    def test_zero_lag(self):
        X = np.array([[10.0], [20.0], [30.0]])
        lt = LagTransformer(lag=0)
        X_lag = lt.fit_transform(X)
        # lag=0 means no shift
        np.testing.assert_allclose(X_lag[:, 0], [10.0, 20.0, 30.0])

    def test_feature_names_out(self):
        X = np.array([[10.0], [20.0], [30.0]])
        lt = LagTransformer(lag=-1)
        lt.fit(X)
        names = lt.get_feature_names_out()
        assert len(names) == 1
        assert "-1" in names[0]

    def test_feature_count_mismatch_raises(self):
        X_train = np.array([[10.0, 20.0], [30.0, 40.0]])
        X_test = np.array([[10.0], [20.0]])
        lt = LagTransformer()
        lt.fit(X_train)
        with pytest.raises(ValueError):
            lt.transform(X_test)

    def test_get_parameter_schema(self):
        schema = LagTransformer.get_parameter_schema()
        assert "lag" in schema
        assert "columns" in schema
        assert "fill_with_missing" in schema
        assert "invert_selection" in schema


# --------------------------------------------------------------------------
# Tests for DifferenceTransformer.
# --------------------------------------------------------------------------

class TestDifferenceTransformer:
    """Tests for DifferenceTransformer."""

    def test_init_defaults(self):
        dt = DifferenceTransformer()
        assert dt.lag == -1
        assert dt.columns is None
        assert dt.fill_with_missing is True
        assert dt.invert_selection is False

    def test_basic_difference_lag_neg1(self):
        X = np.array([[10.0], [12.0], [11.0], [15.0]])
        dt = DifferenceTransformer(lag=-1)
        X_diff = dt.fit_transform(X)
        assert X_diff.shape == (4, 1)
        assert np.isnan(X_diff[0, 0])
        np.testing.assert_allclose(X_diff[1, 0], 2.0)
        np.testing.assert_allclose(X_diff[2, 0], -1.0)
        np.testing.assert_allclose(X_diff[3, 0], 4.0)

    def test_positive_lag(self):
        X = np.array([[10.0], [12.0], [11.0], [15.0]])
        dt = DifferenceTransformer(lag=1)
        X_diff = dt.fit_transform(X)
        assert X_diff.shape == (4, 1)
        # lag=1 means current - future
        np.testing.assert_allclose(X_diff[0, 0], -2.0)
        np.testing.assert_allclose(X_diff[1, 0], 1.0)
        np.testing.assert_allclose(X_diff[2, 0], -4.0)
        assert np.isnan(X_diff[3, 0])

    def test_lag_neg2(self):
        X = np.array([[10.0], [20.0], [30.0], [40.0]])
        dt = DifferenceTransformer(lag=-2)
        X_diff = dt.fit_transform(X)
        assert np.isnan(X_diff[0, 0])
        assert np.isnan(X_diff[1, 0])
        np.testing.assert_allclose(X_diff[2, 0], 20.0)  # 30 - 10
        np.testing.assert_allclose(X_diff[3, 0], 20.0)  # 40 - 20

    def test_fill_with_missing_false(self):
        X = np.array([[10.0], [20.0], [30.0], [40.0]])
        dt = DifferenceTransformer(lag=-1, fill_with_missing=False)
        X_diff = dt.fit_transform(X)
        # First row should be removed (boundary)
        assert X_diff.shape[0] == 3
        np.testing.assert_allclose(X_diff[0, 0], 10.0)

    def test_columns_parameter(self):
        X = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
        dt = DifferenceTransformer(lag=-1, columns=[0])
        X_diff = dt.fit_transform(X)
        assert X_diff.shape == (3, 2)
        # Column 0 should be differenced
        assert np.isnan(X_diff[0, 0])
        np.testing.assert_allclose(X_diff[1, 0], 10.0)
        # Column 1 should be unchanged
        np.testing.assert_allclose(X_diff[:, 1], [100.0, 200.0, 300.0])

    def test_invert_selection(self):
        X = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
        dt = DifferenceTransformer(lag=-1, columns=[0], invert_selection=True)
        X_diff = dt.fit_transform(X)
        # Column 0 should be unchanged (inverted means apply to non-specified)
        np.testing.assert_allclose(X_diff[:, 0], [10.0, 20.0, 30.0])
        # Column 1 should be differenced
        assert np.isnan(X_diff[0, 1])
        np.testing.assert_allclose(X_diff[1, 1], 100.0)

    def test_zero_lag(self):
        X = np.array([[10.0], [20.0], [30.0]])
        dt = DifferenceTransformer(lag=0)
        X_diff = dt.fit_transform(X)
        # lag=0 means current - current = 0
        np.testing.assert_allclose(X_diff[:, 0], [0.0, 0.0, 0.0])

    def test_feature_names_out(self):
        X = np.array([[10.0], [20.0], [30.0]])
        dt = DifferenceTransformer(lag=-1)
        dt.fit(X)
        names = dt.get_feature_names_out()
        assert len(names) == 1
        assert "d-1" in names[0]

    def test_feature_count_mismatch_raises(self):
        X_train = np.array([[10.0, 20.0], [30.0, 40.0]])
        X_test = np.array([[10.0], [20.0]])
        dt = DifferenceTransformer()
        dt.fit(X_train)
        with pytest.raises(ValueError):
            dt.transform(X_test)

    def test_get_parameter_schema(self):
        schema = DifferenceTransformer.get_parameter_schema()
        assert "lag" in schema
        assert "columns" in schema
        assert "fill_with_missing" in schema
        assert "invert_selection" in schema
