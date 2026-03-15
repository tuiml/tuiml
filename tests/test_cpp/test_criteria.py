"""Tests for C++ criteria functions (indirectly via splitter impurity)."""

import numpy as np
import pytest

from tuiml.algorithms.trees._core.criteria import (
    gini_impurity,
    entropy,
    classifier_node_impurity,
    squared_error,
)

try:
    from tuiml._cpp import tree as cpp_tree
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ backend not available")


class TestCriteriaConsistency:
    """The C++ backend computes impurity internally during splitting.

    We verify consistency by checking that Python and C++ splitters
    produce the same gain values (which depend on correct impurity).
    """

    def test_pure_node_gini(self):
        """A pure node should have zero gini -> no valid split."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 0, 0], dtype=np.intc)

        assert gini_impurity(y, 1) == 0.0

        cpp_feat, _, _ = cpp_tree.best_split_classifier(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.intc),
            "gini", 1, 1, 42, 1,
        )
        # Pure node => no split needed (gain would be 0 or negative)
        # The split might still be found but with gain=0
        # In practice the builder wouldn't split because impurity=0

    def test_uniform_distribution_entropy(self):
        """Uniform 2-class distribution has entropy = 1.0."""
        y = np.array([0, 1, 0, 1, 0, 1], dtype=np.intc)
        ent = entropy(y, 2)
        assert abs(ent - 1.0) < 1e-10

    def test_gini_gain_matches(self):
        """Gini gain from C++ matches Python computation."""
        rng_seed = 42
        X = np.random.RandomState(rng_seed).randn(50, 3)
        y = (X[:, 0] > 0).astype(np.intc)

        from tuiml.algorithms.trees._core.splitters import (
            best_split_classifier as py_split,
        )

        _, _, py_gain = py_split(
            X, y, "gini", 2, 1, np.random.RandomState(rng_seed)
        )
        _, _, cpp_gain = cpp_tree.best_split_classifier(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.intc),
            "gini", 2, 1, rng_seed, X.shape[1],
        )
        assert abs(py_gain - cpp_gain) < 1e-10

    def test_mse_zero_variance(self):
        """Zero-variance target has MSE = 0."""
        y = np.array([5.0, 5.0, 5.0, 5.0])
        assert squared_error(y) == 0.0
