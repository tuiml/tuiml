"""Utility functions for the tree engine."""

from __future__ import annotations

import numpy as np
from typing import Optional


def get_tree_description(
    node,
    classes: Optional[np.ndarray] = None,
    depth: int = 0,
    task: str = "classification",
) -> str:
    """Return a human-readable text description of a tree.

    Parameters
    ----------
    node : TreeNode
        Starting node.
    classes : np.ndarray or None
        Class labels (for classifiers).
    depth : int
        Current indentation depth.
    task : str
        ``"classification"`` or ``"regression"``.

    Returns
    -------
    desc : str
        Multi-line text representation.
    """
    indent = "  " * depth
    if node.is_leaf:
        if task == "classification" and classes is not None and node.value is not None:
            pred_class = classes[np.argmax(node.value)]
            return f"{indent}Leaf: class={pred_class} (dist={np.round(node.value, 3)})\n"
        elif node.value is not None:
            return f"{indent}Leaf: value={node.value[0]:.4f}\n"
        elif node.predicted_value is not None:
            return f"{indent}Leaf: value={node.predicted_value:.4f}\n"
        elif node.predicted_class is not None:
            return f"{indent}Leaf: class={node.predicted_class}\n"
        else:
            return f"{indent}Leaf\n"
    desc = f"{indent}Feature[{node.feature_index}] <= {node.threshold:.4f}\n"
    desc += get_tree_description(node.left, classes, depth + 1, task)
    desc += f"{indent}Feature[{node.feature_index}] > {node.threshold:.4f}\n"
    desc += get_tree_description(node.right, classes, depth + 1, task)
    return desc
