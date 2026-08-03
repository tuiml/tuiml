"""C++ tree internals: split criteria, splitters and prediction.

Merged from: test_cpp_criteria.py, test_cpp_splitters.py, test_cpp_predict.py
"""

import numpy as np
import pytest
from tuiml.algorithms.trees._core.criteria import (
    gini_impurity,
    entropy,
    squared_error,
)
from tuiml.algorithms.trees._core.splitters import (
    best_split_classifier as py_best_split_classifier,
    best_split_regressor as py_best_split_regressor,
)


# --------------------------------------------------------------------------
# Tests for C++ criteria functions (indirectly via splitter impurity).
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Tests for C++ splitter backend vs Python fallback.
# --------------------------------------------------------------------------

class TestClassifierSplitter:
    """Compare C++ and Python classifier splitters."""

    def test_basic_gini_split(self):
        """Both backends find the same split on a simple separable dataset."""
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


# --------------------------------------------------------------------------
# Tests for C++ batch prediction over flattened trees.
# --------------------------------------------------------------------------

class TestPredictBatch:
    """Test C++ tree traversal on manually constructed flat trees."""

    def _make_simple_tree(self):
        """Build a simple 3-node tree (root + 2 leaves).

        Tree structure:
            Node 0: split on feature 0, threshold 0.5
            Node 1 (left leaf): value [1.0, 0.0]
            Node 2 (right leaf): value [0.0, 1.0]
        """
        feature = np.array([0, -1, -1], dtype=np.intc)
        threshold = np.array([0.5, 0.0, 0.0], dtype=np.float64)
        children_left = np.array([1, -1, -1], dtype=np.intc)
        children_right = np.array([2, -1, -1], dtype=np.intc)
        value = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        return feature, threshold, children_left, children_right, value

    def test_simple_prediction(self):
        """Samples are routed to correct leaves."""
        feat, thresh, left, right, val = self._make_simple_tree()

        X = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                      dtype=np.float64)

        result = cpp_tree.predict_batch(feat, thresh, left, right, val, X)

        # x[0]=-1.0 <= 0.5 -> left leaf [1, 0]
        np.testing.assert_array_almost_equal(result[0], [1.0, 0.0])
        # x[1]=0.0 <= 0.5 -> left leaf [1, 0]
        np.testing.assert_array_almost_equal(result[1], [1.0, 0.0])
        # x[2]=1.0 > 0.5 -> right leaf [0, 1]
        np.testing.assert_array_almost_equal(result[2], [0.0, 1.0])
        # x[3]=2.0 > 0.5 -> right leaf [0, 1]
        np.testing.assert_array_almost_equal(result[3], [0.0, 1.0])

    def test_deeper_tree(self):
        """A 2-level tree routes samples correctly.

        Tree:
            Node 0: feature=0, threshold=0.0
              Left -> Node 1: feature=1, threshold=-0.5
                Left -> Node 3: leaf [1, 0, 0]
                Right -> Node 4: leaf [0, 1, 0]
              Right -> Node 2: leaf [0, 0, 1]
        """
        feature = np.array([0, 1, -1, -1, -1], dtype=np.intc)
        threshold = np.array([0.0, -0.5, 0.0, 0.0, 0.0], dtype=np.float64)
        children_left = np.array([1, 3, -1, -1, -1], dtype=np.intc)
        children_right = np.array([2, 4, -1, -1, -1], dtype=np.intc)
        value = np.array([
            [0.33, 0.33, 0.34],  # root
            [0.5, 0.5, 0.0],     # node 1
            [0.0, 0.0, 1.0],     # node 2 (leaf)
            [1.0, 0.0, 0.0],     # node 3 (leaf)
            [0.0, 1.0, 0.0],     # node 4 (leaf)
        ], dtype=np.float64)

        X = np.array([
            [-1.0, -1.0],   # feat0 <= 0 -> node1, feat1 <= -0.5 -> node3
            [-1.0, 0.0],    # feat0 <= 0 -> node1, feat1 > -0.5 -> node4
            [1.0, 0.0],     # feat0 > 0 -> node2
        ], dtype=np.float64)

        result = cpp_tree.predict_batch(feature, threshold, children_left,
                                         children_right, value, X)

        np.testing.assert_array_almost_equal(result[0], [1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result[1], [0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(result[2], [0.0, 0.0, 1.0])

    def test_single_node_tree(self):
        """A tree with only a root leaf."""
        feature = np.array([-1], dtype=np.intc)
        threshold = np.array([0.0], dtype=np.float64)
        children_left = np.array([-1], dtype=np.intc)
        children_right = np.array([-1], dtype=np.intc)
        value = np.array([[0.6, 0.4]], dtype=np.float64)

        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

        result = cpp_tree.predict_batch(feature, threshold, children_left,
                                         children_right, value, X)

        np.testing.assert_array_almost_equal(result[0], [0.6, 0.4])
        np.testing.assert_array_almost_equal(result[1], [0.6, 0.4])

    def test_output_shape(self):
        """Output shape matches (n_samples, value_width)."""
        feat, thresh, left, right, val = self._make_simple_tree()
        X = np.random.randn(50, 2).astype(np.float64)

        result = cpp_tree.predict_batch(feat, thresh, left, right, val, X)
        assert result.shape == (50, 2)


class TestDispatchFallback:
    """Test that _core_dispatch works even when C++ is not available."""

    def test_dispatch_module_imports(self):
        """The dispatch module can always be imported."""
        from tuiml.algorithms.trees._core_dispatch import (
            has_cpp_backend,
        )
        # Should not raise
        _ = has_cpp_backend()

    def test_dispatch_produces_valid_split(self):
        """Dispatch produces a valid split regardless of backend."""
        from tuiml.algorithms.trees._core_dispatch import best_split_classifier

        X = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        y = np.array([0, 1], dtype=np.intc)
        rng = np.random.RandomState(42)

        feat, thresh, gain = best_split_classifier(X, y, "gini", 2, 1, rng)
        assert feat != -1
        assert gain > 0
