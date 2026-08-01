"""Tests for RandomOverSampler."""

import numpy as np
import pytest

from tuiml.preprocessing.sampling.oversampling import RandomOverSampler


@pytest.fixture
def imbalanced_data():
    np.random.seed(42)
    X = np.random.randn(60, 2)
    y = np.array([0] * 50 + [1] * 10)
    return X, y


def test_init_defaults():
    sampler = RandomOverSampler()
    assert sampler.sampling_strategy == "auto"
    assert sampler.random_state is None
    assert sampler.shrinkage is None


def test_fit_resample_balances_classes(imbalanced_data):
    X, y = imbalanced_data
    sampler = RandomOverSampler(random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)
    unique, counts = np.unique(y_res, return_counts=True)
    assert counts[0] == counts[1]


def test_output_has_more_samples(imbalanced_data):
    X, y = imbalanced_data
    sampler = RandomOverSampler(random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)
    assert len(X_res) > len(X)


def test_shrinkage_adds_noise(imbalanced_data):
    X, y = imbalanced_data
    sampler = RandomOverSampler(random_state=42, shrinkage=0.1)
    X_res, y_res = sampler.fit_resample(X, y)
    new_samples = X_res[len(X):]
    X_minority = X[y == 1]
    assert len(new_samples) > 0

    # Without shrinkage every new sample is an exact copy of a minority row;
    # with it, each one is jittered. Asserting only that new samples exist
    # would pass either way and so would not test shrinkage at all.
    exact_copies = sum(
        bool(np.any(np.all(np.isclose(X_minority, row), axis=1)))
        for row in new_samples
    )
    assert exact_copies == 0


def test_reproducibility(imbalanced_data):
    X, y = imbalanced_data
    s1 = RandomOverSampler(random_state=42)
    X1, y1 = s1.fit_resample(X, y)
    s2 = RandomOverSampler(random_state=42)
    X2, y2 = s2.fit_resample(X, y)
    np.testing.assert_allclose(X1, X2)


def test_transform_raises():
    sampler = RandomOverSampler()
    with pytest.raises(NotImplementedError):
        sampler.transform(np.array([[1, 2]]))


def test_get_parameter_schema():
    schema = RandomOverSampler.get_parameter_schema()
    assert "sampling_strategy" in schema
    assert "random_state" in schema
    assert "shrinkage" in schema
