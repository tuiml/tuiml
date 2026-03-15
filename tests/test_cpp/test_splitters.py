"""Tests for C++ splitter backend vs Python fallback.

Verifies that the C++ best_split_classifier and best_split_regressor
produce results consistent with the pure-Python implementations.
"""

import numpy as np
import pytest

# Python fallback
from tuiml.algorithms.trees._core.splitters import (
    best_split_classifier as py_best_split_classifier,
    best_split_regressor as py_best_split_regressor,
)

# Try to import C++ backend
try:
    from tuiml._cpp import tree as cpp_tree
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ backend not available")


class TestClassifierSplitter:
    """Compare C++ and Python classifier splitters."""

    def test_basic_gini_split(self):
        """Both backends find the same split on a simple separable dataset."""
        rng = np.random.RandomState(42)
        # Class 0: features < 0, Class 1: features > 0
        X = np.array([[-2.0, 0.1], [-1.0, 0.3], [1.0, -0.2], [2.0, -0.1]])
        y = np.array([0, 0, 1, 1], dtype=np.intc)

        py_feat, py_thresh, py_gain = py_best_split_classifier(
            X, y, "gini", 2, 1, np.random.RandomState(42)
        )

        cpp_feat, cpp_thresh, cpp_gain = cpp_tree.best_split_classifier(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.intc),
            "gini", 2, 1, 42, X.shape[1],
        )

        # Both should find a valid split
        assert py_feat != -1
        assert cpp_feat != -1
        # Gains should be close (may differ slightly due to feature order)
        assert abs(py_gain - cpp_gain) < 1e-10

    def test_entropy_split(self):
        """Entropy criterion produces consistent gains."""
        rng_seed = 123
        X = np.random.RandomState(rng_seed).randn(100, 5)
        y = (X[:, 0] > 0).astype(np.intc)

        py_feat, py_thresh, py_gain = py_best_split_classifier(
            X, y, "entropy", 2, 1, np.random.RandomState(rng_seed)
        )

        cpp_feat, cpp_thresh, cpp_gain = cpp_tree.best_split_classifier(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.intc),
            "entropy", 2, 1, rng_seed, X.shape[1],
        )

        assert py_feat != -1
        assert cpp_feat != -1
        # Gains should be very close
        assert abs(py_gain - cpp_gain) < 1e-8

    def test_no_valid_split(self):
        """Both return -1 when no valid split exists."""
        X = np.array([[1.0], [1.0], [1.0]])
        y = np.array([0, 0, 0], dtype=np.intc)

        py_feat, _, _ = py_best_split_classifier(
            X, y, "gini", 1, 1, np.random.RandomState(0)
        )
        cpp_feat, _, _ = cpp_tree.best_split_classifier(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.intc),
            "gini", 1, 1, 0, 1,
        )
        assert py_feat == -1
        assert cpp_feat == -1

    def test_min_samples_leaf_respected(self):
        """min_samples_leaf is respected by both backends."""
        X = np.array([[-1.0], [0.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1], dtype=np.intc)

        # min_samples_leaf=3 should prevent any valid split with 4 samples
        py_feat, _, _ = py_best_split_classifier(
            X, y, "gini", 2, 3, np.random.RandomState(0)
        )
        cpp_feat, _, _ = cpp_tree.best_split_classifier(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.intc),
            "gini", 2, 3, 0, 1,
        )
        assert py_feat == -1
        assert cpp_feat == -1

    def test_multiclass(self):
        """Multi-class split is consistent."""
        rng_seed = 77
        rng = np.random.RandomState(rng_seed)
        X = rng.randn(200, 10)
        y = (X[:, 0] * 3).astype(int).clip(0, 2).astype(np.intc)

        py_feat, py_thresh, py_gain = py_best_split_classifier(
            X, y, "gini", 3, 5, np.random.RandomState(rng_seed)
        )
        cpp_feat, cpp_thresh, cpp_gain = cpp_tree.best_split_classifier(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.intc),
            "gini", 3, 5, rng_seed, X.shape[1],
        )

        assert py_feat != -1
        assert cpp_feat != -1
        assert abs(py_gain - cpp_gain) < 1e-8


class TestRegressorSplitter:
    """Compare C++ and Python regressor splitters."""

    def test_basic_mse_split(self):
        """Both backends find the same regression split."""
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([-2.0, -1.0, 1.0, 2.0])

        py_feat, py_thresh, py_gain = py_best_split_regressor(
            X, y, "squared_error", 1, np.random.RandomState(42)
        )
        cpp_feat, cpp_thresh, cpp_gain = cpp_tree.best_split_regressor(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            "squared_error", 1, 42, X.shape[1],
        )

        assert py_feat != -1
        assert cpp_feat != -1
        assert abs(py_gain - cpp_gain) < 1e-10

    def test_friedman_mse(self):
        """Friedman MSE criterion is consistent."""
        rng_seed = 99
        X = np.random.RandomState(rng_seed).randn(100, 5)
        y = X[:, 0] * 2.0 + X[:, 1] + np.random.RandomState(rng_seed).randn(100) * 0.1

        py_feat, py_thresh, py_gain = py_best_split_regressor(
            X, y, "friedman_mse", 5, np.random.RandomState(rng_seed)
        )
        cpp_feat, cpp_thresh, cpp_gain = cpp_tree.best_split_regressor(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            "friedman_mse", 5, rng_seed, X.shape[1],
        )

        assert py_feat != -1
        assert cpp_feat != -1
        assert abs(py_gain - cpp_gain) < 1e-8

    def test_no_valid_split_regressor(self):
        """Both return -1 when all feature values are identical."""
        X = np.ones((5, 2))
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        py_feat, _, _ = py_best_split_regressor(
            X, y, "squared_error", 1, np.random.RandomState(0)
        )
        cpp_feat, _, _ = cpp_tree.best_split_regressor(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            "squared_error", 1, 0, 2,
        )
        assert py_feat == -1
        assert cpp_feat == -1
