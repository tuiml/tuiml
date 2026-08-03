"""Scaling filters.

Merged from: test_scaling_standardize.py, test_scaling_normalize.py, test_scaling_center.py
"""

import numpy as np
import pytest
from tuiml.preprocessing.scaling.standardize import StandardScaler
from tuiml.preprocessing.scaling.normalize import MinMaxScaler
from tuiml.preprocessing.scaling.center import CenterScaler


# --------------------------------------------------------------------------
# Tests for StandardScaler transformer.
# --------------------------------------------------------------------------

class TestStandardScaler:
    """Tests for StandardScaler transformer."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return np.random.randn(100, 3) * 10 + 5

    def test_init_defaults(self):
        scaler = StandardScaler()
        assert scaler.with_mean is True
        assert scaler.with_std is True
        assert scaler.columns is None

    def test_fit_transform_zero_mean_unit_std(self, sample_data):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sample_data)
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_scaled.std(axis=0), 1.0, atol=1e-10)

    def test_inverse_transform_recovers_original(self, sample_data):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sample_data)
        X_recovered = scaler.inverse_transform(X_scaled)
        np.testing.assert_allclose(X_recovered, sample_data, atol=1e-10)

    def test_with_mean_false(self, sample_data):
        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(sample_data)
        # Mean should NOT be zero when with_mean=False
        assert not np.allclose(X_scaled.mean(axis=0), 0.0, atol=0.1)
        # But std should still be 1
        np.testing.assert_allclose(X_scaled.std(axis=0), 1.0, atol=1e-10)

    def test_with_std_false(self, sample_data):
        scaler = StandardScaler(with_std=False)
        X_scaled = scaler.fit_transform(sample_data)
        # Mean should be zero
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
        # But std should NOT be 1 (unless it happened to be already)
        original_std = sample_data.std(axis=0)
        np.testing.assert_allclose(X_scaled.std(axis=0), original_std, atol=1e-10)

    def test_columns_parameter(self):
        np.random.seed(42)
        X = np.array([[1.0, 100.0, 10.0],
                      [2.0, 200.0, 20.0],
                      [3.0, 300.0, 30.0]])
        scaler = StandardScaler(columns=[1])
        X_scaled = scaler.fit_transform(X)
        # Column 0 and 2 should be unchanged
        np.testing.assert_allclose(X_scaled[:, 0], X[:, 0])
        np.testing.assert_allclose(X_scaled[:, 2], X[:, 2])
        # Column 1 should be standardized
        np.testing.assert_allclose(X_scaled[:, 1].mean(), 0.0, atol=1e-10)

    def test_get_parameter_schema(self):
        schema = StandardScaler.get_parameter_schema()
        assert "with_mean" in schema
        assert "with_std" in schema
        assert "columns" in schema

    def test_zero_variance_column(self):
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Zero variance column should not produce NaN or inf
        assert not np.any(np.isnan(X_scaled))
        assert not np.any(np.isinf(X_scaled))

    def test_feature_count_mismatch_raises(self):
        scaler = StandardScaler()
        X_train = np.array([[1, 2], [3, 4]])
        scaler.fit(X_train)
        X_test = np.array([[1, 2, 3]])
        with pytest.raises(ValueError):
            scaler.transform(X_test)


# --------------------------------------------------------------------------
# Tests for MinMaxScaler transformer.
# --------------------------------------------------------------------------

class TestMinMaxScaler:
    """Tests for MinMaxScaler transformer."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return np.random.randn(100, 3) * 10 + 5

    def test_init_defaults(self):
        scaler = MinMaxScaler()
        assert scaler.scale == 1.0
        assert scaler.translation == 0.0
        assert scaler.columns is None

    def test_fit_transform_range_01(self, sample_data):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(sample_data)
        np.testing.assert_allclose(X_scaled.min(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_scaled.max(axis=0), 1.0, atol=1e-10)

    def test_custom_range(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        scaler = MinMaxScaler(scale=2.0, translation=-1.0)
        X_scaled = scaler.fit_transform(X)
        np.testing.assert_allclose(X_scaled.min(axis=0), -1.0, atol=1e-10)
        np.testing.assert_allclose(X_scaled.max(axis=0), 1.0, atol=1e-10)

    def test_inverse_transform_recovers_original(self, sample_data):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(sample_data)
        X_recovered = scaler.inverse_transform(X_scaled)
        np.testing.assert_allclose(X_recovered, sample_data, atol=1e-10)

    def test_columns_parameter(self):
        X = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        scaler = MinMaxScaler(columns=[0])
        X_scaled = scaler.fit_transform(X)
        # Column 0 should be in [0, 1]
        np.testing.assert_allclose(X_scaled[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(X_scaled[2, 0], 1.0, atol=1e-10)
        # Column 1 should be unchanged
        np.testing.assert_allclose(X_scaled[:, 1], X[:, 1])

    def test_get_parameter_schema(self):
        schema = MinMaxScaler.get_parameter_schema()
        assert "scale" in schema
        assert "translation" in schema
        assert "columns" in schema

    def test_zero_range_column(self):
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        assert not np.any(np.isnan(X_scaled))
        assert not np.any(np.isinf(X_scaled))

    def test_feature_count_mismatch_raises(self):
        scaler = MinMaxScaler()
        X_train = np.array([[1, 2], [3, 4]])
        scaler.fit(X_train)
        X_test = np.array([[1, 2, 3]])
        with pytest.raises(ValueError):
            scaler.transform(X_test)


# --------------------------------------------------------------------------
# Tests for CenterScaler transformer.
# --------------------------------------------------------------------------

class TestCenterScaler:
    """Tests for CenterScaler transformer."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return np.random.randn(100, 3) * 10 + 5

    def test_init_defaults(self):
        scaler = CenterScaler()
        assert scaler.columns is None

    def test_fit_transform_zero_mean(self, sample_data):
        scaler = CenterScaler()
        X_centered = scaler.fit_transform(sample_data)
        np.testing.assert_allclose(X_centered.mean(axis=0), 0.0, atol=1e-10)

    def test_std_unchanged(self, sample_data):
        scaler = CenterScaler()
        original_std = sample_data.std(axis=0)
        X_centered = scaler.fit_transform(sample_data)
        np.testing.assert_allclose(X_centered.std(axis=0), original_std, atol=1e-10)

    def test_inverse_transform_recovers_original(self, sample_data):
        scaler = CenterScaler()
        X_centered = scaler.fit_transform(sample_data)
        X_recovered = scaler.inverse_transform(X_centered)
        np.testing.assert_allclose(X_recovered, sample_data, atol=1e-10)

    def test_columns_parameter(self):
        X = np.array([[1.0, 100.0, 10.0],
                      [2.0, 200.0, 20.0],
                      [3.0, 300.0, 30.0]])
        scaler = CenterScaler(columns=[1])
        X_centered = scaler.fit_transform(X)
        # Column 0 and 2 should be unchanged
        np.testing.assert_allclose(X_centered[:, 0], X[:, 0])
        np.testing.assert_allclose(X_centered[:, 2], X[:, 2])
        # Column 1 should be centered
        np.testing.assert_allclose(X_centered[:, 1].mean(), 0.0, atol=1e-10)

    def test_get_parameter_schema(self):
        schema = CenterScaler.get_parameter_schema()
        assert "columns" in schema

    def test_feature_count_mismatch_raises(self):
        scaler = CenterScaler()
        X_train = np.array([[1, 2], [3, 4]])
        scaler.fit(X_train)
        X_test = np.array([[1, 2, 3]])
        with pytest.raises(ValueError):
            scaler.transform(X_test)
