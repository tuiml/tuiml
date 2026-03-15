"""Tests for VarianceThresholdSelector."""

import numpy as np
import pytest

from tuiml.features.selection import VarianceThresholdSelector


@pytest.fixture
def sample_data():
    """Create sample data with known variance characteristics."""
    np.random.seed(42)
    # Column 0: constant (variance = 0)
    # Column 1: low variance
    # Column 2: high variance
    # Column 3: constant (variance = 0)
    n = 50
    col_const1 = np.zeros(n)                         # constant 0
    col_low = np.random.uniform(0, 0.01, n)          # very low variance
    col_high = np.random.randn(n) * 10               # high variance
    col_const2 = np.zeros(n)                          # constant 0
    X = np.column_stack([col_const1, col_low, col_high, col_const2])
    return X


class TestVarianceThresholdSelectorInit:

    def test_default_init(self):
        selector = VarianceThresholdSelector()
        assert selector.threshold == 0.0

    def test_custom_threshold(self):
        selector = VarianceThresholdSelector(threshold=0.5)
        assert selector.threshold == 0.5

    def test_variances_none_before_fit(self):
        selector = VarianceThresholdSelector()
        assert selector.variances_ is None


class TestVarianceThresholdSelectorFit:

    def test_removes_constant_features(self, sample_data):
        selector = VarianceThresholdSelector(threshold=0.0)
        X_new = selector.fit_transform(sample_data)
        # Columns 0 and 3 are constant, should be removed
        assert X_new.shape[1] == 2

    def test_threshold_parameter(self, sample_data):
        # With a higher threshold, more features should be removed
        selector_low = VarianceThresholdSelector(threshold=0.0)
        selector_high = VarianceThresholdSelector(threshold=1.0)

        X_low = selector_low.fit_transform(sample_data)
        X_high = selector_high.fit_transform(sample_data)

        assert X_high.shape[1] <= X_low.shape[1]

    def test_fit_stores_variances(self, sample_data):
        selector = VarianceThresholdSelector()
        selector.fit(sample_data)
        assert selector.variances_ is not None
        assert len(selector.variances_) == sample_data.shape[1]

    def test_negative_threshold_raises(self):
        selector = VarianceThresholdSelector(threshold=-1.0)
        X = np.random.randn(10, 3)
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            selector.fit(X)


class TestVarianceThresholdSelectorTransform:

    def test_fit_transform(self, sample_data):
        selector = VarianceThresholdSelector(threshold=0.0)
        X_new = selector.fit_transform(sample_data)
        assert X_new.shape[0] == sample_data.shape[0]
        assert X_new.shape[1] < sample_data.shape[1]

    def test_transform_before_fit_raises(self):
        selector = VarianceThresholdSelector()
        X = np.random.randn(10, 3)
        with pytest.raises(RuntimeError):
            selector.transform(X)

    def test_all_constant_features(self):
        X = np.ones((20, 5))
        selector = VarianceThresholdSelector()
        X_new = selector.fit_transform(X)
        assert X_new.shape[1] == 0

    def test_no_features_removed_when_all_vary(self):
        np.random.seed(42)
        X = np.random.randn(50, 4) * 5
        selector = VarianceThresholdSelector(threshold=0.0)
        X_new = selector.fit_transform(X)
        assert X_new.shape[1] == 4


class TestVarianceThresholdSelectorSupport:

    def test_get_support_mask(self, sample_data):
        selector = VarianceThresholdSelector(threshold=0.0)
        selector.fit(sample_data)
        mask = selector.get_support(indices=False)
        assert mask.dtype == bool
        assert len(mask) == sample_data.shape[1]
        # Constant columns (0 and 3) should be False
        assert mask[0] is np.bool_(False)
        assert mask[3] is np.bool_(False)

    def test_get_support_indices(self, sample_data):
        selector = VarianceThresholdSelector(threshold=0.0)
        selector.fit(sample_data)
        indices = selector.get_support(indices=True)
        assert indices.dtype in [np.int64, np.int32, np.intp]
        # Should contain indices 1 and 2 (non-constant columns)
        assert 1 in indices
        assert 2 in indices
        assert 0 not in indices
        assert 3 not in indices

    def test_get_support_before_fit_raises(self):
        selector = VarianceThresholdSelector()
        with pytest.raises(RuntimeError):
            selector.get_support()


class TestVarianceThresholdSelectorSchema:

    def test_get_parameter_schema(self):
        schema = VarianceThresholdSelector.get_parameter_schema()
        assert "threshold" in schema
        assert schema["threshold"]["type"] == "number"
        assert schema["threshold"]["default"] == 0.0
