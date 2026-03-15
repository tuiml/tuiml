"""Tests for IQROutlierDetector transformer."""

import numpy as np
import pytest

from tuiml.preprocessing.outliers.interquartile_range import IQROutlierDetector


@pytest.fixture
def data_with_outliers():
    np.random.seed(42)
    X = np.random.randn(100, 2)
    # Add extreme outliers
    X[0, 0] = 100.0
    X[1, 1] = -100.0
    return X


def test_init_defaults():
    detector = IQROutlierDetector()
    assert detector.factor == 1.5
    assert detector.action == "clip"
    assert detector.columns is None


def test_fit_returns_self(data_with_outliers):
    detector = IQROutlierDetector()
    result = detector.fit(data_with_outliers)
    assert result is detector


def test_clip_action(data_with_outliers):
    detector = IQROutlierDetector(action="clip")
    X_clean = detector.fit_transform(data_with_outliers)
    # The extreme outlier should be clipped
    assert X_clean[0, 0] < 100.0
    assert X_clean[1, 1] > -100.0


def test_nan_action(data_with_outliers):
    detector = IQROutlierDetector(action="nan")
    X_clean = detector.fit_transform(data_with_outliers)
    # Outliers should be replaced with NaN
    assert np.isnan(X_clean[0, 0])
    assert np.isnan(X_clean[1, 1])


def test_remove_action(data_with_outliers):
    detector = IQROutlierDetector(action="remove")
    X_clean = detector.fit_transform(data_with_outliers)
    # Should have fewer rows
    assert X_clean.shape[0] < data_with_outliers.shape[0]


def test_factor_parameter():
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


def test_bounds_property(data_with_outliers):
    detector = IQROutlierDetector()
    detector.fit(data_with_outliers)
    bounds = detector.bounds_
    assert 0 in bounds
    assert 1 in bounds
    lower, upper = bounds[0]
    assert lower < upper


def test_invalid_action_raises():
    with pytest.raises(ValueError):
        IQROutlierDetector(action="invalid")


def test_transform_before_fit_raises():
    detector = IQROutlierDetector()
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(RuntimeError):
        detector.transform(X)


def test_get_parameter_schema():
    schema = IQROutlierDetector.get_parameter_schema()
    assert "factor" in schema
    assert "action" in schema
    assert "columns" in schema
