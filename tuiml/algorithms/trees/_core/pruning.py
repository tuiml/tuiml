"""Pruning strategies for decision trees."""

from __future__ import annotations

import numpy as np
from typing import Optional

from .nodes import TreeNode


def populate_node_stats(
    node: TreeNode,
    X: np.ndarray,
    y: np.ndarray,
    criterion: str,
    n_classes: Optional[int] = None,
) -> None:
    """Fill in ``n_samples`` and ``impurity`` on every node of a built tree.

    Parameters
    ----------
    node : TreeNode
        Root of the tree to annotate, modified in place.
    X : np.ndarray of shape (n_samples, n_features)
        Training data the tree was built from.
    y : np.ndarray of shape (n_samples,)
        Training targets; integer-encoded labels for a classifier.
    criterion : str
        Splitting criterion, used to compute each node's impurity.
    n_classes : int or None, default=None
        Number of classes for a classifier tree. ``None`` selects the
        regression impurity.

    Notes
    -----
    The C++ tree builder returns nodes with ``n_samples`` and ``impurity``
    left at zero, because it only tracks what prediction needs. Cost-complexity
    pruning needs both, and dividing by an unpopulated ``n_samples`` raises
    ``ZeroDivisionError``. Calling this first makes a C++-built tree carry the
    same statistics as a Python-built one.
    """
    from .criteria import classifier_node_impurity, regressor_node_impurity

    stack = [(node, np.arange(len(y)))]
    while stack:
        nd, idx = stack.pop()
        nd.n_samples = int(len(idx))
        y_sub = y[idx]
        if len(idx) == 0:
            nd.impurity = 0.0
        elif n_classes is None:
            nd.impurity = float(regressor_node_impurity(y_sub, criterion))
        else:
            nd.impurity = float(
                classifier_node_impurity(y_sub, criterion, n_classes)
            )
        if not nd.is_leaf and nd.feature_index >= 0:
            left_mask = X[idx, nd.feature_index] <= nd.threshold
            if nd.left is not None:
                stack.append((nd.left, idx[left_mask]))
            if nd.right is not None:
                stack.append((nd.right, idx[~left_mask]))


def cost_complexity_prune(node: TreeNode, ccp_alpha: float) -> TreeNode:
    """Apply minimal cost-complexity pruning (CART).

    Parameters
    ----------
    node : TreeNode
        Root of the (sub)tree to prune.
    ccp_alpha : float
        Complexity parameter. Subtrees with effective alpha less
        than ``ccp_alpha`` are pruned.

    Returns
    -------
    node : TreeNode
        Pruned node (may become a leaf).
    """
    if node.is_leaf:
        return node

    node.left = cost_complexity_prune(node.left, ccp_alpha)
    node.right = cost_complexity_prune(node.right, ccp_alpha)

    if node.left.is_leaf and node.right.is_leaf:
        n_total = node.n_samples
        leaf_impurity = node.impurity
        left_imp = (node.left.n_samples / n_total) * node.left.impurity
        right_imp = (node.right.n_samples / n_total) * node.right.impurity
        subtree_impurity = left_imp + right_imp

        n_leaves = 2
        alpha = (leaf_impurity - subtree_impurity) / (n_leaves - 1)
        if alpha <= ccp_alpha:
            return TreeNode(
                is_leaf=True,
                value=node.value,
                n_samples=node.n_samples,
                impurity=node.impurity,
            )

    return node
