"""Universal tree node and flattened tree data structures."""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TreeNode:
    """Universal tree node used by all tree algorithms.

    Parameters
    ----------
    is_leaf : bool
        Whether this is a terminal node.
    feature_index : int
        Feature used for the split (``-1`` for leaves).
    threshold : float
        Split threshold (``0.0`` for leaves).
    left : TreeNode or None
        Left child (samples where feature <= threshold).
    right : TreeNode or None
        Right child (samples where feature > threshold).
    n_samples : int
        Number of training samples at this node.
    impurity : float
        Impurity at this node before splitting.
    value : np.ndarray or None
        Leaf value: class distribution (classifier) or ``[mean]`` (regressor).
    linear_model : np.ndarray or None
        M5P: regression coefficients at leaf.
    intercept : float
        M5P: intercept term.
    weights : np.ndarray or None
        LMT: logistic regression weights.
    bias : Any
        LMT: logistic regression bias.
    class_distribution : dict or None
        Class distribution dict for backward compat.
    predicted_class : Any
        Predicted class index for backward compat.
    predicted_value : float or None
        Predicted regression value for backward compat.
    is_numeric : bool
        Whether the split feature is numeric (for C4.5 nominal support).
    """

    is_leaf: bool = False
    feature_index: int = -1
    threshold: float = 0.0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    n_samples: int = 0
    impurity: float = 0.0
    value: Optional[np.ndarray] = None

    # M5P fields
    linear_model: Optional[np.ndarray] = None
    intercept: float = 0.0

    # LMT fields
    weights: Optional[np.ndarray] = None
    bias: Any = 0.0

    # Backward compat fields
    class_distribution: Optional[Dict] = None
    predicted_class: Any = None
    predicted_value: Optional[float] = None

    # C4.5 nominal support
    is_numeric: bool = True


@dataclass
class FlattenedTree:
    """Flat array representation of a decision tree for fast traversal.

    Each node is stored at an index in parallel arrays. Internal nodes store
    the indices of their children; leaves store ``-1``.

    Parameters
    ----------
    feature : array of shape (n_nodes,)
        Feature index for each node (``-1`` for leaves).
    threshold : array of shape (n_nodes,)
        Split threshold for each node.
    children_left : array of shape (n_nodes,)
        Index of the left child (``-1`` for leaves).
    children_right : array of shape (n_nodes,)
        Index of the right child (``-1`` for leaves).
    value : array of shape (n_nodes, value_width)
        Node values -- class distribution (classifier) or ``[[mean]]`` (regressor).
    n_nodes : int
        Total number of nodes.
    """

    feature: Any = None
    threshold: Any = None
    children_left: Any = None
    children_right: Any = None
    value: Any = None
    n_nodes: int = 0


def flatten_tree(root: TreeNode, value_width: int) -> FlattenedTree:
    """Convert a recursive ``TreeNode`` tree to flat NumPy arrays for prediction.

    Parameters
    ----------
    root : TreeNode
        Root of the recursively built tree.
    value_width : int
        Width of the value array (``n_classes`` for classifiers, ``1``
        for regressors).

    Returns
    -------
    flat : FlattenedTree
        Flattened representation with NumPy arrays.
    """
    features: List[int] = []
    thresholds: List[float] = []
    left_children: List[int] = []
    right_children: List[int] = []
    values: List[np.ndarray] = []

    node_index = 0

    def _add_node(node: TreeNode) -> int:
        nonlocal node_index
        idx = node_index
        node_index += 1
        features.append(node.feature_index if not node.is_leaf else -1)
        thresholds.append(node.threshold if not node.is_leaf else 0.0)
        left_children.append(-1)
        right_children.append(-1)
        val = node.value if node.value is not None else np.zeros(value_width)
        if val.ndim == 0:
            val = np.array([val])
        padded = np.zeros(value_width)
        padded[: len(val)] = val[:value_width]
        values.append(padded)
        return idx

    root_idx = _add_node(root)
    stack: List[Tuple[TreeNode, int]] = [(root, root_idx)]

    while stack:
        node, idx = stack.pop()
        if node.is_leaf:
            continue
        if node.left is not None:
            left_idx = _add_node(node.left)
            left_children[idx] = left_idx
            stack.append((node.left, left_idx))
        if node.right is not None:
            right_idx = _add_node(node.right)
            right_children[idx] = right_idx
            stack.append((node.right, right_idx))

    return FlattenedTree(
        feature=np.array(features, dtype=np.int32),
        threshold=np.array(thresholds, dtype=np.float32),
        children_left=np.array(left_children, dtype=np.int32),
        children_right=np.array(right_children, dtype=np.int32),
        value=np.array(np.array(values), dtype=np.float32),
        n_nodes=node_index,
    )


def count_nodes(node: Optional[TreeNode]) -> int:
    """Count total nodes in a ``TreeNode`` tree.

    Parameters
    ----------
    node : TreeNode or None
        Root node.

    Returns
    -------
    count : int
        Number of nodes.
    """
    if node is None:
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)


def max_depth_of(node: Optional[TreeNode]) -> int:
    """Compute the actual depth of a ``TreeNode`` tree.

    Parameters
    ----------
    node : TreeNode or None
        Root node.

    Returns
    -------
    depth : int
        Maximum depth (root = 0).
    """
    if node is None or node.is_leaf:
        return 0
    return 1 + max(max_depth_of(node.left), max_depth_of(node.right))
