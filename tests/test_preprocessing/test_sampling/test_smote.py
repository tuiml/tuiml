"""Tests for SMOTESampler and variants."""

import numpy as np
import pytest

from tuiml.preprocessing.sampling.smote import SMOTESampler


@pytest.fixture
def imbalanced_data():
    np.random.seed(42)
    X_majority = np.random.randn(50, 2) + 5
    X_minority = np.random.randn(10, 2)
    X = np.vstack([X_majority, X_minority])
    y = np.array([0] * 50 + [1] * 10)
    return X, y


def test_init_defaults():
    sampler = SMOTESampler()
    assert sampler.sampling_strategy == "auto"
    assert sampler.k_neighbors == 5
    assert sampler.random_state is None


def test_fit_resample_balances_classes(imbalanced_data):
    X, y = imbalanced_data
    sampler = SMOTESampler(k_neighbors=3, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)
    # After SMOTE, minority should match majority
    unique, counts = np.unique(y_res, return_counts=True)
    assert counts[0] == counts[1]


def test_output_has_more_samples(imbalanced_data):
    X, y = imbalanced_data
    sampler = SMOTESampler(k_neighbors=3, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)
    assert len(X_res) > len(X)
    assert len(y_res) > len(y)


def test_original_data_preserved(imbalanced_data):
    X, y = imbalanced_data
    sampler = SMOTESampler(k_neighbors=3, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)
    # Original data should be in the beginning
    np.testing.assert_allclose(X_res[:len(X)], X)


def test_reproducibility(imbalanced_data):
    X, y = imbalanced_data
    s1 = SMOTESampler(k_neighbors=3, random_state=42)
    X1, y1 = s1.fit_resample(X, y)
    s2 = SMOTESampler(k_neighbors=3, random_state=42)
    X2, y2 = s2.fit_resample(X, y)
    np.testing.assert_allclose(X1, X2)
    np.testing.assert_allclose(y1, y2)


def test_transform_raises():
    sampler = SMOTESampler()
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(NotImplementedError):
        sampler.transform(X)


def test_get_parameter_schema():
    schema = SMOTESampler.get_parameter_schema()
    assert "sampling_strategy" in schema
    assert "k_neighbors" in schema
    assert "random_state" in schema


def test_too_few_minority_samples_raises():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 0, 0, 0, 1])  # Only 1 minority sample, k=5
    sampler = SMOTESampler(k_neighbors=5)
    with pytest.raises(ValueError):
        sampler.fit_resample(X, y)
