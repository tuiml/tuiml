"""Tests for RandomUnderSampler."""

import numpy as np
import pytest

from tuiml.preprocessing.sampling.undersampling import RandomUnderSampler


@pytest.fixture
def imbalanced_data():
    np.random.seed(42)
    X = np.random.randn(60, 2)
    y = np.array([0] * 50 + [1] * 10)
    return X, y


def test_init_defaults():
    sampler = RandomUnderSampler()
    assert sampler.sampling_strategy == "auto"
    assert sampler.random_state is None
    assert sampler.replacement is False


def test_fit_resample_balances_classes(imbalanced_data):
    X, y = imbalanced_data
    sampler = RandomUnderSampler(random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)
    unique, counts = np.unique(y_res, return_counts=True)
    # Majority should be reduced to match minority
    assert counts[0] == counts[1]


def test_output_has_fewer_samples(imbalanced_data):
    X, y = imbalanced_data
    sampler = RandomUnderSampler(random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)
    assert len(X_res) < len(X)


def test_minority_class_preserved(imbalanced_data):
    X, y = imbalanced_data
    sampler = RandomUnderSampler(random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)
    # All minority samples should still be present
    minority_count = np.sum(y_res == 1)
    assert minority_count == 10


def test_reproducibility(imbalanced_data):
    X, y = imbalanced_data
    s1 = RandomUnderSampler(random_state=42)
    X1, y1 = s1.fit_resample(X, y)
    s2 = RandomUnderSampler(random_state=42)
    X2, y2 = s2.fit_resample(X, y)
    np.testing.assert_allclose(X1, X2)


def test_transform_raises():
    sampler = RandomUnderSampler()
    with pytest.raises(NotImplementedError):
        sampler.transform(np.array([[1, 2]]))


def test_get_parameter_schema():
    schema = RandomUnderSampler.get_parameter_schema()
    assert "sampling_strategy" in schema
    assert "random_state" in schema
    assert "replacement" in schema
