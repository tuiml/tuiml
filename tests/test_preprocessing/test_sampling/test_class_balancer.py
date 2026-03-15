"""Tests for ClassBalanceSampler."""

import numpy as np
import pytest
from collections import Counter

from tuiml.preprocessing.sampling.class_balancer import ClassBalanceSampler


@pytest.fixture
def imbalanced_data():
    np.random.seed(42)
    X = np.arange(100).reshape(-1, 1).astype(float)
    y = np.array([0] * 90 + [1] * 10)
    return X, y


def test_init_defaults():
    balancer = ClassBalanceSampler()
    assert balancer.strategy == "oversample"
    assert balancer.target_ratio == 1.0
    assert balancer.random_state is None


def test_oversample_strategy(imbalanced_data):
    X, y = imbalanced_data
    balancer = ClassBalanceSampler(strategy="oversample", random_state=42)
    balancer.fit(X)
    X_bal, y_bal = balancer.transform(X, y)
    counts = Counter(y_bal)
    assert counts[0] == counts[1]


def test_undersample_strategy(imbalanced_data):
    X, y = imbalanced_data
    balancer = ClassBalanceSampler(strategy="undersample", random_state=42)
    balancer.fit(X)
    X_bal, y_bal = balancer.transform(X, y)
    counts = Counter(y_bal)
    assert counts[0] == counts[1]


def test_both_strategy(imbalanced_data):
    X, y = imbalanced_data
    balancer = ClassBalanceSampler(strategy="both", random_state=42)
    balancer.fit(X)
    X_bal, y_bal = balancer.transform(X, y)
    counts = Counter(y_bal)
    assert counts[0] == counts[1]


def test_fit_returns_self(imbalanced_data):
    X, y = imbalanced_data
    balancer = ClassBalanceSampler()
    result = balancer.fit(X)
    assert result is balancer


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        ClassBalanceSampler(strategy="invalid")


def test_invalid_target_ratio_raises():
    with pytest.raises(ValueError):
        ClassBalanceSampler(target_ratio=0.0)
    with pytest.raises(ValueError):
        ClassBalanceSampler(target_ratio=2.0)


def test_transform_before_fit_raises():
    balancer = ClassBalanceSampler()
    X = np.array([[1], [2]])
    y = np.array([0, 1])
    with pytest.raises(RuntimeError):
        balancer.transform(X, y)


def test_get_parameter_schema():
    schema = ClassBalanceSampler.get_parameter_schema()
    assert "strategy" in schema
    assert "target_ratio" in schema
    assert "random_state" in schema
