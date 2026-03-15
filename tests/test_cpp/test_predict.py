"""Tests for C++ batch prediction over flattened trees."""

import numpy as np
import pytest

try:
    from tuiml._cpp import tree as cpp_tree
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ backend not available")


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
            best_split_classifier,
            best_split_regressor,
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
