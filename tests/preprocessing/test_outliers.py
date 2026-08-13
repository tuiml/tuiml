"""Outlier handling filters.

Covers value clipping and IQR outlier detection.
"""

import numpy as np
from tuiml.preprocessing.outliers.clip import ValueClipper
import pytest
from tuiml.preprocessing.outliers.iqr import IQROutlierDetector


# --------------------------------------------------------------------------
# Tests for ValueClipper transformer.
# --------------------------------------------------------------------------

class TestValueClipper:
    """Tests for ValueClipper transformer."""

    def test_init_defaults(self):
        clipper = ValueClipper()
        assert clipper.lower is None
        assert clipper.upper is None
        assert clipper.percentile is None
        assert clipper.columns is None

    def test_fixed_bounds(self):
        X = np.array([[-10], [5], [20]])
        clipper = ValueClipper(lower=0, upper=10)
        X_clipped = clipper.fit_transform(X)
        np.testing.assert_allclose(X_clipped.flatten(), [0.0, 5.0, 10.0])

    def test_lower_only(self):
        X = np.array([[-10], [5], [20]])
        clipper = ValueClipper(lower=0)
        X_clipped = clipper.fit_transform(X)
        np.testing.assert_allclose(X_clipped.flatten(), [0.0, 5.0, 20.0])

    def test_upper_only(self):
        X = np.array([[-10], [5], [20]])
        clipper = ValueClipper(upper=10)
        X_clipped = clipper.fit_transform(X)
        np.testing.assert_allclose(X_clipped.flatten(), [-10.0, 5.0, 10.0])

    def test_percentile_bounds(self):
        np.random.seed(42)
        X = np.random.randn(1000, 1)
        clipper = ValueClipper(percentile=(1, 99))
        X_clipped = clipper.fit_transform(X)
        assert X_clipped.min() >= np.percentile(X, 1) - 1e-10
        assert X_clipped.max() <= np.percentile(X, 99) + 1e-10

    def test_columns_parameter(self):
        X = np.array([[-10.0, -10.0], [5.0, 5.0], [20.0, 20.0]])
        clipper = ValueClipper(lower=0, upper=10, columns=[0])
        X_clipped = clipper.fit_transform(X)
        np.testing.assert_allclose(X_clipped[:, 0], [0.0, 5.0, 10.0])
        # Column 1 should be unchanged
        np.testing.assert_allclose(X_clipped[:, 1], [-10.0, 5.0, 20.0])

    def test_get_parameter_schema(self):
        schema = ValueClipper.get_parameter_schema()
        assert "lower" in schema
        assert "upper" in schema
        assert "percentile" in schema
        assert "columns" in schema


# --------------------------------------------------------------------------
# Tests for IQROutlierDetector transformer.
# --------------------------------------------------------------------------

class TestIQROutlierDetector:
    """Tests for IQROutlierDetector transformer."""

    @pytest.fixture
    def data_with_outliers(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        # Add extreme outliers
        X[0, 0] = 100.0
        X[1, 1] = -100.0
        return X

    def test_init_defaults(self):
        detector = IQROutlierDetector()
        assert detector.factor == 1.5
        assert detector.action == "clip"
        assert detector.columns is None

    def test_clip_action(self, data_with_outliers):
        detector = IQROutlierDetector(action="clip")
        X_clean = detector.fit_transform(data_with_outliers)
        # The extreme outlier should be clipped
        assert X_clean[0, 0] < 100.0
        assert X_clean[1, 1] > -100.0

    def test_nan_action(self, data_with_outliers):
        detector = IQROutlierDetector(action="nan")
        X_clean = detector.fit_transform(data_with_outliers)
        # Outliers should be replaced with NaN
        assert np.isnan(X_clean[0, 0])
        assert np.isnan(X_clean[1, 1])

    def test_remove_action(self, data_with_outliers):
        detector = IQROutlierDetector(action="remove")
        X_clean = detector.fit_transform(data_with_outliers)
        # Should have fewer rows
        assert X_clean.shape[0] < data_with_outliers.shape[0]

    def test_factor_parameter(self):
        np.random.seed(42)
        X = np.random.randn(100, 1)
        X[0, 0] = 5.0  # Mild outlier

        # With factor 1.5, should catch it
        detector_mild = IQROutlierDetector(factor=1.5, action="nan")
        X_mild = detector_mild.fit_transform(X)

        # With factor 3.0, might not catch it
        detector_extreme = IQROutlierDetector(factor=3.0, action="nan")
        X_extreme = detector_extreme.fit_transform(X)

        mild_nans = np.sum(np.isnan(X_mild))
        extreme_nans = np.sum(np.isnan(X_extreme))
        assert mild_nans >= extreme_nans

    def test_bounds_property(self, data_with_outliers):
        detector = IQROutlierDetector()
        detector.fit(data_with_outliers)
        bounds = detector.bounds_
        assert 0 in bounds
        assert 1 in bounds
        lower, upper = bounds[0]
        assert lower < upper

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError):
            IQROutlierDetector(action="invalid")

    def test_get_parameter_schema(self):
        schema = IQROutlierDetector.get_parameter_schema()
        assert "factor" in schema
        assert "action" in schema
        assert "columns" in schema
