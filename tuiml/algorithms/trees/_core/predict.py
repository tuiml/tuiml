"""NumPy-based prediction engine for flattened and recursive tree structures."""

from __future__ import annotations

import numpy as np

from .nodes import TreeNode, FlattenedTree


def build_jit_functions():
    """Backward-compatible no-op retained for older imports."""
    return None


def _predict_single_flat(flat_tree: FlattenedTree, x: np.ndarray) -> np.ndarray:
    """Traverse a flattened tree for one sample."""
    node_idx = 0
    while flat_tree.children_left[node_idx] != -1:
        feat = flat_tree.feature[node_idx]
        if x[feat] <= flat_tree.threshold[node_idx]:
            node_idx = flat_tree.children_left[node_idx]
        else:
            node_idx = flat_tree.children_right[node_idx]
    return flat_tree.value[node_idx]


def predict_batch(flat_tree: FlattenedTree, X: np.ndarray) -> np.ndarray:
    """Predict over a batch of samples using flattened NumPy tree arrays."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return np.array([_predict_single_flat(flat_tree, x) for x in X], dtype=np.float32)


def predict_proba_batch(flat_tree: FlattenedTree, X: np.ndarray) -> np.ndarray:
    """Predict class probabilities for a batch of samples."""
    return predict_batch(flat_tree, X)


def predict_single_numpy(node: TreeNode, x: np.ndarray) -> np.ndarray:
    """Traverse a recursive tree for one sample (NumPy)."""
    while not node.is_leaf:
        feat_val = x[node.feature_index]
        if np.isnan(feat_val):
            if node.left is not None and (
                node.right is None or node.left.n_samples >= node.right.n_samples
            ):
                node = node.left
            else:
                node = node.right
        elif feat_val <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value


def predict_proba_single_numpy(
    node: TreeNode, x: np.ndarray, n_classes: int
) -> np.ndarray:
    """Predict class probabilities for one sample (NumPy)."""
    val = predict_single_numpy(node, x)
    if val is None:
        return np.ones(n_classes) / n_classes
    return val
