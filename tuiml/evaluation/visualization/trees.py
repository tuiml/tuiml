"""
Decision tree visualization.

Renders tree structures from all TuiML tree algorithm classes
as matplotlib figures with sklearn-style node boxes, edges, and labels.

Uses bigtree library for Reingold-Tilford layout computation,
then draws annotated nodes with matplotlib.

Also provides text-based exports similar to scikit-learn:
- export_text: ASCII text representation of the tree
- export_graphviz: DOT format for GraphViz rendering
"""

from io import StringIO
from typing import List, Optional, Tuple, Union, Dict, Any
from numbers import Integral
import warnings

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.text import Annotation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    Annotation = None

try:
    from bigtree import Node as BTNode
    from bigtree import reingold_tilford
    HAS_BIGTREE = True
except ImportError:
    HAS_BIGTREE = False
    BTNode = None

from ._style import apply_style, PALETTES

# Node colors from default palette
_COLORS = {
    'internal': '#4C72B0',   # Steel blue
    'leaf_cls': '#55A868',   # Sage green
    'leaf_reg': '#DD8452',   # Coral/orange
    'truncated': '#9CA3AF',  # Gray
}


def _color_brew(n):
    """Generate n colors with equally spaced hues.

    Parameters
    ----------
    n : int
        The number of colors required.

    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360.0 / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.0
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [
            (c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0),
        ]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))), (int(255 * (g + m))), (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list


class _TreeNodeAdapter:
    """Unified wrapper around any TuiML tree node.

    Auto-detects node type by checking attributes and provides a
    consistent interface for the layout and drawing code.

    Parameters
    ----------
    node : object
        A tree node from any TuiML tree algorithm.
    """

    def __init__(self, node):
        self._node = node

    @property
    def is_leaf(self):
        return getattr(self._node, 'is_leaf', False)

    @property
    def feature_index(self):
        # M5 uses split_attr
        if hasattr(self._node, 'split_attr'):
            return self._node.split_attr
        return getattr(self._node, 'feature_index', None)

    @property
    def threshold(self):
        # M5 uses split_value
        if hasattr(self._node, 'split_value'):
            return self._node.split_value
        return getattr(self._node, 'threshold', None)

    @property
    def left(self):
        child = getattr(self._node, 'left', None)
        return _TreeNodeAdapter(child) if child is not None else None

    @property
    def right(self):
        child = getattr(self._node, 'right', None)
        return _TreeNodeAdapter(child) if child is not None else None

    @property
    def children(self):
        """For nominal (multi-way) splits like C4.5."""
        ch = getattr(self._node, 'children', None)
        if ch and isinstance(ch, dict):
            return {k: _TreeNodeAdapter(v) for k, v in ch.items()}
        return None

    @property
    def is_nominal_split(self):
        node = self._node
        if getattr(node, 'is_numeric', True) is False:
            return True
        if hasattr(node, 'children') and isinstance(getattr(node, 'children', None), dict):
            if not self.is_leaf and getattr(node, 'children', None):
                return True
        return False

    @property
    def n_samples(self):
        return getattr(self._node, 'n_samples', None)

    @property
    def is_classification(self):
        node = self._node
        if hasattr(node, 'predicted_class'):
            return True
        if hasattr(node, 'class_distribution'):
            return True
        if hasattr(node, 'class_counts'):
            return True
        if hasattr(node, 'weights') and hasattr(node, 'bias'):
            return True
        return False

    def get_split_label(self, feature_names=None):
        """Return a human-readable split condition string."""
        fi = self.feature_index
        if fi is None:
            return '?'
        fname = feature_names[fi] if feature_names and fi < len(feature_names) else f'X[{fi}]'
        th = self.threshold
        if th is None:
            return fname
        if isinstance(th, float):
            return f'{fname} <= {th:.3g}'
        return f'{fname} <= {th}'

    def get_prediction_label(self, class_names=None):
        """Return a human-readable prediction string for leaf nodes."""
        node = self._node

        # Classification nodes
        if hasattr(node, 'predicted_class'):
            pc = node.predicted_class
            if class_names is not None:
                try:
                    pc = class_names[int(pc)]
                except (IndexError, ValueError, TypeError):
                    pass
            label = str(pc)
        elif hasattr(node, 'class_counts'):
            # Hoeffding tree - pick majority class
            counts = node.class_counts
            if counts:
                majority = max(counts, key=counts.get)
                if class_names is not None:
                    try:
                        majority = class_names[int(majority)]
                    except (IndexError, ValueError, TypeError):
                        pass
                label = str(majority)
            else:
                label = '?'
        elif hasattr(node, 'predicted_value'):
            label = f'{node.predicted_value:.3g}'
        elif hasattr(node, 'prediction'):
            label = f'{node.prediction:.3g}'
        elif hasattr(node, 'linear_model'):
            label = 'LM'
        else:
            label = '?'

        # Append sample count if available
        ns = self.n_samples
        if ns is not None:
            label += f'\nn={ns}'

        return label

    def get_node_color(self):
        """Return the appropriate color for this node."""
        if self.is_leaf:
            if self.is_classification:
                return _COLORS['leaf_cls']
            return _COLORS['leaf_reg']
        return _COLORS['internal']


# ---------------------------------------------------------------------------
# Bigtree-based visualization helpers
# ---------------------------------------------------------------------------

def _build_node_label(adapter, feature_names=None, class_names=None,
                      label_mode="all", precision=3, depth=0):
    """Build a multi-line sklearn-style text label for a tree node.

    Parameters
    ----------
    adapter : _TreeNodeAdapter
        The node to label.
    feature_names : list of str, optional
        Feature names.
    class_names : list of str, optional
        Class names.
    label_mode : str
        One of 'all', 'root', 'none'.
    precision : int
        Decimal precision for floats.
    depth : int
        Current depth (used with label_mode='root').

    Returns
    -------
    label : str
        Multi-line label string.
    """
    lines = []
    show_labels = (label_mode == "root" and depth == 0) or label_mode == "all"

    # Split condition (internal nodes only)
    if not adapter.is_leaf:
        fi = adapter.feature_index
        if fi is not None:
            fname = (feature_names[fi]
                     if feature_names and fi < len(feature_names)
                     else f'X[{fi}]')
            th = adapter.threshold
            if th is not None:
                if isinstance(th, float):
                    lines.append(f'{fname} <= {th:.{precision}g}')
                else:
                    lines.append(f'{fname} <= {th}')
            else:
                lines.append(fname)

    # Sample count
    if show_labels:
        n_samp = adapter.n_samples
        if n_samp is not None:
            lines.append(f'samples = {n_samp}')

    # Leaf prediction
    if adapter.is_leaf:
        pred = adapter.get_prediction_label(class_names)
        if '\n' in pred:
            pred = pred.split('\n')[0]
        if adapter.is_classification:
            lines.append(f'class = {pred}')
        else:
            lines.append(f'value = {pred}')

    return '\n'.join(lines) if lines else '?'


def _build_bigtree_nodes(adapter, feature_names=None, class_names=None,
                         max_depth=None, depth=0, counter=None,
                         label_mode="all", precision=3):
    """Convert a TuiML adapter tree into a bigtree Node tree.

    Parameters
    ----------
    adapter : _TreeNodeAdapter
        Root adapter node.

    Returns
    -------
    bt_node : bigtree.Node
        Root of the bigtree structure with layout-ready attributes.
    """
    if counter is None:
        counter = [0]

    node_id = counter[0]
    counter[0] += 1

    # Truncation check
    truncated = (not adapter.is_leaf
                 and max_depth is not None
                 and depth >= max_depth)

    if truncated:
        text = '(...)'
    else:
        text = _build_node_label(
            adapter, feature_names, class_names,
            label_mode, precision, depth,
        )

    bt_node = BTNode(
        str(node_id),
        node_label=text,
        is_leaf_node=adapter.is_leaf or truncated,
        is_truncated=truncated,
        is_classification=adapter.is_classification,
        node_adapter=adapter,
        edge_label=None,
    )

    # Recurse into children (unless leaf or truncated)
    if not adapter.is_leaf and not truncated:
        if adapter.is_nominal_split and adapter.children:
            for branch_label, child_adapter in adapter.children.items():
                child_bt = _build_bigtree_nodes(
                    child_adapter, feature_names, class_names,
                    max_depth, depth + 1, counter, label_mode, precision,
                )
                child_bt.parent = bt_node
                child_bt.edge_label = str(branch_label)
        else:
            if adapter.left:
                left_bt = _build_bigtree_nodes(
                    adapter.left, feature_names, class_names,
                    max_depth, depth + 1, counter, label_mode, precision,
                )
                left_bt.parent = bt_node
                left_bt.edge_label = "True"
            if adapter.right:
                right_bt = _build_bigtree_nodes(
                    adapter.right, feature_names, class_names,
                    max_depth, depth + 1, counter, label_mode, precision,
                )
                right_bt.parent = bt_node
                right_bt.edge_label = "False"

    return bt_node


def _iter_nodes(bt_node):
    """Pre-order iteration over bigtree nodes."""
    yield bt_node
    for child in bt_node.children:
        yield from _iter_nodes(child)


def _get_node_facecolor(node, filled):
    """Determine face color for a bigtree node."""
    if not filled:
        return 'white'
    if node.is_truncated:
        return _COLORS['truncated']
    return node.node_adapter.get_node_color()


def _compute_box_size(label, fontsize):
    """Compute box width and height for a node label at given fontsize."""
    lines = label.split('\n')
    max_line_len = max((len(l) for l in lines), default=1)
    n_lines = len(lines)
    char_w = fontsize * 0.009
    line_h = fontsize * 0.032
    box_w = max(1.2, max_line_len * char_w + 0.4)
    box_h = max(0.4, n_lines * line_h + 0.25)
    return box_w, box_h


def _draw_bigtree(bt_root, ax, filled=False, rounded=True, fontsize=None):
    """Draw bigtree nodes as sklearn-style boxes connected by edges.

    Parameters
    ----------
    bt_root : bigtree.Node
        Root node (must have x, y from reingold_tilford).
    ax : matplotlib.axes.Axes
        Target axes.
    filled : bool
        Whether to fill node boxes with color.
    rounded : bool
        Whether to use rounded box corners.
    fontsize : int or None
        Font size; auto-computed if None.
    """
    nodes = list(_iter_nodes(bt_root))
    if not nodes:
        return

    # Auto font size based on tree complexity
    if fontsize is None:
        n_nodes = len(nodes)
        fontsize = max(8, min(12, 14 - n_nodes // 5))

    # Pre-compute box sizes for all nodes
    box_sizes = {}
    for node in nodes:
        bw, bh = _compute_box_size(node.node_label, fontsize)
        box_sizes[id(node)] = (bw, bh)

    max_bw = max(bw for bw, _ in box_sizes.values())
    max_bh = max(bh for _, bh in box_sizes.values())

    # Scale bigtree coordinates so boxes don't overlap.
    # bigtree sibling_separation=1.0 means adjacent siblings are 1.0 apart.
    # We scale x so the gap equals max_box_width + padding.
    x_scale = max_bw * 1.3
    y_scale = max_bh * 1.8
    for node in nodes:
        node.x *= x_scale
        node.y *= y_scale

    # Collect final coordinates
    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]

    # --- Draw edges (underneath everything) ---
    for node in nodes:
        if node.parent is not None:
            # Draw line from parent bottom to child top
            p_bh = box_sizes[id(node.parent)][1]
            c_bh = box_sizes[id(node)][1]
            ax.plot(
                [node.parent.x, node.x],
                [node.parent.y - p_bh / 2, node.y + c_bh / 2],
                color='#444444', linewidth=1.0,
                solid_capstyle='round', zorder=1,
            )
            # Edge labels on root's immediate children only
            edge_label = node.edge_label
            if edge_label and node.parent.is_root:
                mid_x = (node.x + node.parent.x) / 2
                mid_y = (node.parent.y - p_bh / 2 + node.y + c_bh / 2) / 2
                is_left = node.x <= node.parent.x
                ha = 'right' if is_left else 'left'
                ax.text(
                    mid_x, mid_y, f'  {edge_label}  ',
                    ha=ha, va='center',
                    fontsize=fontsize,
                    fontweight='bold', color='black', zorder=5,
                )

    # --- Draw node boxes and text ---
    for node in nodes:
        label = node.node_label
        bw, bh = box_sizes[id(node)]
        facecolor = _get_node_facecolor(node, filled)
        edgecolor = '#333333'
        if filled and facecolor not in ('white', _COLORS['truncated']):
            text_color = 'white'
        else:
            text_color = 'black'

        boxstyle = ('round,pad=0.12,rounding_size=0.15'
                     if rounded else 'square,pad=0.12')

        bbox = FancyBboxPatch(
            (node.x - bw / 2, node.y - bh / 2),
            bw, bh,
            boxstyle=boxstyle,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.2,
            alpha=0.95,
            zorder=3,
        )
        ax.add_patch(bbox)

        ax.text(
            node.x, node.y, label,
            ha='center', va='center',
            fontsize=fontsize,
            fontweight='normal',
            color=text_color,
            zorder=4,
            linespacing=1.3,
            fontfamily='sans-serif',
        )

    # Set axis limits with padding
    margin_x = max_bw
    margin_y = max_bh
    ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
    ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)


# ---------------------------------------------------------------------------
# Tree size utilities
# ---------------------------------------------------------------------------

def _count_leaves(adapter, max_depth=None, depth=0):
    """Count leaf nodes (or truncation points) in the tree."""
    if adapter.is_leaf:
        return 1
    if max_depth is not None and depth >= max_depth:
        return 1

    # Handle nominal (multi-way) splits
    if adapter.is_nominal_split and adapter.children:
        total = 0
        for child in adapter.children.values():
            total += _count_leaves(child, max_depth, depth + 1)
        return total

    left_count = _count_leaves(adapter.left, max_depth, depth + 1) if adapter.left else 0
    right_count = _count_leaves(adapter.right, max_depth, depth + 1) if adapter.right else 0
    return left_count + right_count


def _tree_depth(adapter, max_depth=None, depth=0):
    """Compute the depth of the tree."""
    if adapter.is_leaf:
        return depth
    if max_depth is not None and depth >= max_depth:
        return depth

    if adapter.is_nominal_split and adapter.children:
        return max(
            _tree_depth(child, max_depth, depth + 1)
            for child in adapter.children.values()
        )

    left_d = _tree_depth(adapter.left, max_depth, depth + 1) if adapter.left else depth
    right_d = _tree_depth(adapter.right, max_depth, depth + 1) if adapter.right else depth
    return max(left_d, right_d)


def _check_is_fitted(model):
    """Check if a model is fitted."""
    # Check for common fitted indicators
    if hasattr(model, 'tree_') and model.tree_ is not None:
        return True
    if hasattr(model, 'estimators_') and model.estimators_:
        return True
    if hasattr(model, 'feature_index_') and model.feature_index_ is not None:
        return True
    # Check sklearn-style fitted indicator
    if hasattr(model, 'n_features_in_'):
        return True
    return False


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def plot_tree(
    decision_tree,
    *,
    max_depth=None,
    feature_names=None,
    class_names=None,
    label="all",
    filled=False,
    impurity=True,
    node_ids=False,
    proportion=False,
    rounded=False,
    precision=3,
    ax=None,
    fontsize=None,
    # TuiML-specific parameters (not in sklearn)
    tree_index=0,
    figsize=None,
    save_path=None,
    title=None,
):
    """Plot a decision tree.

    The sample counts that are shown are weighted with any sample_weights that
    might be present.

    The visualization is fit automatically to the size of the axis.
    Use the ``figsize`` or ``dpi`` arguments of ``plt.figure`` to control
    the size of the rendering.

    Parameters
    ----------
    decision_tree : decision tree regressor or classifier
        The decision tree to be plotted. Can be a TuiML tree model
        with a ``tree_`` attribute, a RandomForest with ``estimators_``,
        or a DecisionStump.

    max_depth : int, default=None
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : array-like of str, default=None
        Names of each of the features.
        If None, generic names will be used ("X[0]", "X[1]", ...).

    class_names : array-like of str or bool, default=None
        Names of each of the target classes in ascending numerical order.
        Only relevant for classification and not supported for multi-output.
        If ``True``, shows a symbolic representation of the class name.

    label : {'all', 'root', 'none'}, default='all'
        Whether to show informative labels for impurity, etc.
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.

    filled : bool, default=False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    impurity : bool, default=True
        When set to ``True``, show the impurity at each node.
        Note: Not all TuiML tree models provide impurity information.

    node_ids : bool, default=False
        When set to ``True``, show the ID number on each node.

    proportion : bool, default=False
        When set to ``True``, change the display of 'values' and/or 'samples'
        to be proportions and percentages respectively.

    rounded : bool, default=False
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    precision : int, default=3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.

    ax : matplotlib axis, default=None
        Axes to plot to. If None, use current axis. Any previous content
        is cleared.

    fontsize : int, default=None
        Size of text font. If None, determined automatically to fit figure.

    tree_index : int, default=0
        For RandomForest models, which tree to visualize (TuiML extension).

    figsize : tuple, default=None
        Figure size (width, height) in inches (TuiML extension).
        If None, auto-computed based on tree size.

    save_path : str, default=None
        Path to save the figure as PNG (TuiML extension).

    title : str, default=None
        Custom plot title (TuiML extension).

    Returns
    -------
    annotations : list of artists
        List containing the artists for the annotation boxes making up the
        tree.

    Examples
    --------
    >>> from tuiml.tree import DecisionTreeClassifier
    >>> from tuiml.evaluation.visualization import plot_tree

    >>> clf = DecisionTreeClassifier(random_state=0)
    >>> clf = clf.fit(X, y)
    >>> plot_tree(clf, feature_names=['a', 'b', 'c'])
    [...]
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    if not HAS_BIGTREE:
        raise ImportError(
            "bigtree is required for tree visualization. "
            "Install it with: pip install bigtree"
        )

    # Check if fitted
    if not _check_is_fitted(decision_tree):
        raise ValueError("The tree model is not fitted yet.")

    # Dispatch: RandomForest ensemble
    if hasattr(decision_tree, 'estimators_'):
        estimators = decision_tree.estimators_
        if tree_index >= len(estimators):
            raise ValueError(
                f'tree_index={tree_index} but model has '
                f'{len(estimators)} estimators'
            )
        subtree_model = estimators[tree_index]
        if hasattr(subtree_model, 'tree_'):
            tree_root = subtree_model.tree_
        else:
            tree_root = subtree_model
        default_title = f'Random Forest - Tree {tree_index}'

    # Dispatch: DecisionStump (no tree_ attribute)
    elif hasattr(decision_tree, 'feature_index_') and not hasattr(decision_tree, 'tree_'):
        # Use legacy visualization for DecisionStump
        return _plot_decision_stump_legacy(
            decision_tree, feature_names, class_names, figsize, save_path,
            title, fontsize, filled, rounded
        )

    # Dispatch: standard tree with tree_ attribute
    elif hasattr(decision_tree, 'tree_'):
        tree_root = decision_tree.tree_
        algo_name = type(decision_tree).__name__
        default_title = f'{algo_name}'
    else:
        # Assume it's a tree node itself
        tree_root = decision_tree
        default_title = 'Decision Tree'

    # Build adapter and bigtree node structure
    adapter = _TreeNodeAdapter(tree_root)
    bt_root = _build_bigtree_nodes(
        adapter, feature_names, class_names,
        max_depth=max_depth, label_mode=label, precision=precision,
    )

    # Compute layout using bigtree's Reingold-Tilford algorithm
    reingold_tilford(
        bt_root,
        sibling_separation=1.0,
        subtree_separation=1.5,
        level_separation=2.0,
    )

    # Auto-compute figsize if not provided
    if ax is None:
        if figsize is None:
            n_leaves = _count_leaves(adapter, max_depth)
            depth = _tree_depth(adapter, max_depth)
            width = max(10, n_leaves * 3)
            height = max(6, (depth + 1) * 2.5)
            figsize = (width, height)

        apply_style()
        fig, ax = plt.subplots(figsize=figsize)

    ax.set_axis_off()

    # Draw the tree
    _draw_bigtree(bt_root, ax, filled=filled, rounded=rounded, fontsize=fontsize)

    if title or default_title:
        ax.set_title(
            title or default_title,
            fontsize=14, fontweight='semibold', pad=12,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)

    plt.show()
    return []


# ---------------------------------------------------------------------------
# DecisionStump legacy visualization
# ---------------------------------------------------------------------------

def _plot_decision_stump_legacy(
    model, feature_names, class_names, figsize, save_path,
    title, fontsize, filled, rounded
):
    """Plot a DecisionStump using the bigtree pipeline for consistent layout."""
    apply_style()

    fi = model.feature_index_
    th = model.threshold_
    fname = feature_names[fi] if feature_names and fi < len(feature_names) else f'X[{fi}]'

    # --- Build root label ---
    if getattr(model, 'is_numeric_', True):
        root_lines = [f'{fname} <= {th:.3g}']
    else:
        root_lines = [f'{fname} = {th}']
    total_n = getattr(model, 'n_samples_', None)
    left_n = getattr(model, 'left_n_samples_', None)
    right_n = getattr(model, 'right_n_samples_', None)
    if total_n is None and left_n is not None and right_n is not None:
        total_n = left_n + right_n
    if total_n is not None:
        root_lines.append(f'samples = {total_n}')
    root_text = '\n'.join(root_lines)

    # --- Build left leaf label ---
    left_cls = model.left_class_
    if class_names is not None:
        try:
            left_cls = class_names[int(left_cls)]
        except (IndexError, ValueError, TypeError):
            pass
    left_lines = []
    if left_n is not None:
        left_lines.append(f'samples = {left_n}')
    left_dist = getattr(model, 'left_class_distribution_', None)
    if left_dist is not None:
        left_lines.append(f'value = {list(left_dist)}')
    left_lines.append(f'class = {left_cls}')
    left_text = '\n'.join(left_lines)

    # --- Build right leaf label ---
    right_cls = model.right_class_
    if class_names is not None:
        try:
            right_cls = class_names[int(right_cls)]
        except (IndexError, ValueError, TypeError):
            pass
    right_lines = []
    if right_n is not None:
        right_lines.append(f'samples = {right_n}')
    right_dist = getattr(model, 'right_class_distribution_', None)
    if right_dist is not None:
        right_lines.append(f'value = {list(right_dist)}')
    right_lines.append(f'class = {right_cls}')
    right_text = '\n'.join(right_lines)

    # --- Build 3-node bigtree and use standard pipeline ---
    class _StumpAdapter:
        def __init__(self, is_leaf, is_cls):
            self.is_leaf = is_leaf
            self.is_classification = is_cls
        def get_node_color(self):
            if self.is_leaf:
                return _COLORS['leaf_cls'] if self.is_classification else _COLORS['leaf_reg']
            return _COLORS['internal']

    bt_root = BTNode(
        "0", node_label=root_text, is_leaf_node=False,
        is_truncated=False, is_classification=True,
        node_adapter=_StumpAdapter(False, True), edge_label=None,
    )
    BTNode(
        "1", node_label=left_text, is_leaf_node=True,
        is_truncated=False, is_classification=True,
        node_adapter=_StumpAdapter(True, True), edge_label="True",
        parent=bt_root,
    )
    BTNode(
        "2", node_label=right_text, is_leaf_node=True,
        is_truncated=False, is_classification=True,
        node_adapter=_StumpAdapter(True, True), edge_label="False",
        parent=bt_root,
    )

    reingold_tilford(
        bt_root,
        sibling_separation=1.0,
        subtree_separation=1.5,
        level_separation=2.0,
    )

    if figsize is None:
        figsize = (8, 5)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    _draw_bigtree(bt_root, ax, filled=filled, rounded=rounded, fontsize=fontsize)

    ax.set_title(
        title or 'Decision Stump',
        fontsize=14, fontweight='semibold', pad=12,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)

    plt.show()
    return []


# ---------------------------------------------------------------------------
# Text export
# ---------------------------------------------------------------------------

def export_text(
    decision_tree,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    max_depth: int = 10,
    spacing: int = 3,
    decimals: int = 2,
    show_weights: bool = False,
) -> str:
    """Build a text report showing the rules of a decision tree.

    This is similar to scikit-learn's export_text function and provides
    a human-readable ASCII representation of the tree structure.

    Parameters
    ----------
    decision_tree : fitted tree model
        A fitted TuiML tree algorithm instance with a ``tree_`` attribute,
        or the tree root node itself.
    feature_names : list of str, optional
        Feature names for display. If None, uses ``feature_{i}`` format.
    class_names : list of str, optional
        Class names for classification trees. If None for classification,
        uses the tree's internal class representation.
    max_depth : int, default=10
        Only the first max_depth levels of the tree are exported.
        Truncated branches will be marked with "...".
    spacing : int, default=3
        Number of spaces between edges. The higher it is, the wider the result.
    decimals : int, default=2
        Number of decimal digits to display for thresholds and values.
    show_weights : bool, default=False
        If True for classification trees, the class distribution
        (number of samples per class) will be exported on each leaf.

    Returns
    -------
    report : str
        Text summary of all the rules in the decision tree.

    Examples
    --------
    >>> from tuiml.tree import DecisionTreeClassifier
    >>> from tuiml.evaluation.visualization import export_text
    >>> clf = DecisionTreeClassifier().fit(X, y)
    >>> print(export_text(clf, feature_names=['a', 'b']))
    |--- a <= 0.50
    |   |--- class: 0
    |--- a >  0.50
    |   |--- b <= 1.00
    |   |   |--- class: 1
    |   |--- b >  1.00
    |   |   |--- class: 2
    """
    # Get the tree root
    tree_root = _get_tree_root(decision_tree)
    if tree_root is None:
        raise ValueError(
            "decision_tree must have a tree_ attribute or be a tree node"
        )

    adapter = _TreeNodeAdapter(tree_root)

    report = StringIO()

    # Determine if classification
    is_classifier = adapter.is_classification

    def _indent_str(depth):
        """Create the indentation string for a given depth."""
        if depth == 0:
            return ""
        return "|" + " " * spacing + ("|" + " " * spacing) * (depth - 1) + "-" * spacing

    def _recurse(node_adapter, depth):
        """Recursively traverse the tree and write to report."""
        if depth > max_depth:
            report.write(_indent_str(depth) + "... (truncated)\n")
            return

        if node_adapter.is_leaf:
            # Leaf node - write prediction
            if is_classifier:
                # Get class label
                label = node_adapter.get_prediction_label(class_names)
                # Extract just the class name (remove n=... if present)
                if '\n' in label:
                    label = label.split('\n')[0]

                if show_weights:
                    # Show class distribution if available
                    n_samp = node_adapter.n_samples
                    weights_info = f"n={n_samp}" if n_samp is not None else ""
                    report.write(_indent_str(depth) + f"class: {label}  [{weights_info}]\n")
                else:
                    report.write(_indent_str(depth) + f"class: {label}\n")
            else:
                # Regression - show value
                label = node_adapter.get_prediction_label(class_names)
                if '\n' in label:
                    label = label.split('\n')[0]
                report.write(_indent_str(depth) + f"value: {label}\n")
            return

        # Internal node - write split condition
        fi = node_adapter.feature_index
        th = node_adapter.threshold

        if fi is None:
            # Cannot determine split, just recurse
            _recurse(node_adapter.left, depth + 1)
            _recurse(node_adapter.right, depth + 1)
            return

        # Get feature name
        if feature_names is not None and fi < len(feature_names):
            fname = feature_names[fi]
        else:
            fname = f"feature_{fi}"

        # Format threshold
        if isinstance(th, float):
            th_str = f"{th:.{decimals}f}"
        else:
            th_str = str(th)

        # Write split condition (left branch: <=)
        report.write(_indent_str(depth) + f"{fname} <= {th_str}\n")
        _recurse(node_adapter.left, depth + 1)

        # Write right branch condition (>)
        report.write(_indent_str(depth) + f"{fname} >  {th_str}\n")
        _recurse(node_adapter.right, depth + 1)

    _recurse(adapter, 0)
    return report.getvalue()


# ---------------------------------------------------------------------------
# GraphViz DOT export
# ---------------------------------------------------------------------------

def export_graphviz(
    decision_tree,
    out_file: Optional[Union[str, StringIO]] = None,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    label: str = "all",
    filled: bool = False,
    leaves_parallel: bool = False,
    impurity: bool = True,
    node_ids: bool = False,
    proportion: bool = False,
    rotate: bool = False,
    rounded: bool = False,
    special_characters: bool = False,
    precision: int = 3,
    fontname: str = "helvetica",
    max_depth: Optional[int] = None,
) -> Optional[str]:
    """Export a decision tree in DOT format.

    This function generates a GraphViz representation of the decision tree,
    which can be rendered using the GraphViz ``dot`` tool::

        $ dot -Tpng tree.dot -o tree.png

    Parameters
    ----------
    decision_tree : fitted tree model
        A fitted TuiML tree algorithm instance with a ``tree_`` attribute,
        or the tree root node itself.
    out_file : str or file-like object, optional
        Handle or name of the output file. If ``None``, the result is
        returned as a string.
    feature_names : list of str, optional
        Feature names for display. If None, uses ``feature_{i}`` format.
    class_names : list of str, optional
        Class names for classification trees.
    label : {'all', 'root', 'none'}, default='all'
        Whether to show informative labels (sample counts, etc.).
        Options include 'all' to show at every node, 'root' to show only at
        the top root node, or 'none' to not show at any node.
    filled : bool, default=False
        When set to ``True``, paint nodes to indicate leaf type
        (classification vs regression).
    leaves_parallel : bool, default=False
        When set to ``True``, draw all leaf nodes at the bottom of the tree.
    impurity : bool, default=True
        When set to ``True``, show the impurity (where available) at each node.
    node_ids : bool, default=False
        When set to ``True``, show the ID number on each node.
    proportion : bool, default=False
        When set to ``True``, change the display of sample counts
        to be proportions of the total.
    rotate : bool, default=False
        When set to ``True``, orient tree left to right rather than top-down.
    rounded : bool, default=False
        When set to ``True``, draw node boxes with rounded corners.
    special_characters : bool, default=False
        When set to ``True``, use special characters (e.g., Greek letters)
        for certain symbols.
    precision : int, default=3
        Number of digits of precision for floating point values.
    fontname : str, default='helvetica'
        Name of font used to render text.
    max_depth : int, optional
        Maximum depth of the tree to export. If None, exports the entire tree.

    Returns
    -------
    dot_data : str or None
        String representation of the input tree in GraphViz dot format.
        Only returned if ``out_file`` is None.

    Examples
    --------
    >>> from tuiml.tree import DecisionTreeClassifier
    >>> from tuiml.evaluation.visualization import export_graphviz
    >>> clf = DecisionTreeClassifier().fit(X, y)
    >>> dot_data = export_graphviz(clf, feature_names=['a', 'b'])
    >>> print(dot_data)
    digraph Tree {
    node [shape=box, style="filled", color="black", fontname="helvetica"] ;
    ...
    """
    # Get tree root
    tree_root = _get_tree_root(decision_tree)
    if tree_root is None:
        raise ValueError(
            "decision_tree must have a tree_ attribute or be a tree node"
        )

    adapter = _TreeNodeAdapter(tree_root)

    # Setup output
    return_string = False
    own_file = False

    if out_file is None:
        return_string = True
        out_file = StringIO()
    elif isinstance(out_file, str):
        out_file = open(out_file, "w", encoding="utf-8")
        own_file = True

    try:
        # Write DOT header
        out_file.write("digraph Tree {\n")

        # Node style
        node_style = ["shape=box"]
        if filled:
            node_style.append('style="filled"')
        if rounded:
            node_style.append('style="rounded,filled"' if filled else 'style="rounded"')
        node_style.append(f'fontname="{fontname}"')
        node_style.append('color="black"')

        out_file.write("node [" + ", ".join(node_style) + "] ;\n")
        out_file.write(f'edge [fontname="{fontname}"] ;\n')

        if leaves_parallel:
            out_file.write("graph [ranksep=equally, splines=polyline] ;\n")

        if rotate:
            out_file.write("rankdir=LR ;\n")

        # Track leaf ranks if needed
        leaf_ranks = [] if leaves_parallel else None

        # Node counter for IDs
        node_counter = [0]

        def _get_node_id():
            nid = node_counter[0]
            node_counter[0] += 1
            return nid

        def _format_label(node_adapt, depth):
            """Format the label for a node."""
            lines = []

            # Node ID if requested
            if node_ids:
                lines.append(f"node #{node_counter[0]}")

            if node_adapt.is_leaf:
                # Leaf node
                lab = node_adapt.get_prediction_label(class_names)
                # Clean up for display
                if '\n' in lab:
                    lab = lab.replace('\n', '\\n')

                if node_adapt.is_classification:
                    lines.append(f"class = {lab}")
                else:
                    lines.append(f"value = {lab}")

                # Sample count
                n_samp = node_adapt.n_samples
                if n_samp is not None:
                    lines.append(f"samples = {n_samp}")

            else:
                # Internal node - split condition
                fi = node_adapt.feature_index
                th = node_adapt.threshold

                if fi is not None:
                    if feature_names is not None and fi < len(feature_names):
                        fname = feature_names[fi]
                    else:
                        fname = f"X[{fi}]"

                    if isinstance(th, float):
                        th_str = f"{th:.{precision}g}"
                    else:
                        th_str = str(th)

                    # Use proper comparison symbols
                    if special_characters:
                        lines.append(f"{fname} \u2264 {th_str}")
                    else:
                        lines.append(f"{fname} <= {th_str}")

                # Sample count
                n_samp = node_adapt.n_samples
                if n_samp is not None and label in ("all", "root"):
                    if label == "all" or depth == 0:
                        lines.append(f"samples = {n_samp}")

            return "\\n".join(lines) if lines else ""

        def _get_fill_color(node_adapt):
            """Get fill color for a node."""
            if not filled:
                return None
            return node_adapt.get_node_color()

        def _recurse_dot(node_adapt, parent_id=None, edge_label=None, depth=0):
            """Recursively write nodes and edges in DOT format."""
            if max_depth is not None and depth > max_depth:
                # Truncated node
                nid = _get_node_id()
                lab = "(...)"
                color_attr = f', fillcolor="#C0C0C0"' if filled else ""
                out_file.write(f'{nid} [label="{lab}"{color_attr}] ;\n')

                if parent_id is not None:
                    out_file.write(f"{parent_id} -> {nid}")
                    if edge_label:
                        out_file.write(f' [label="{edge_label}"]')
                    out_file.write(" ;\n")

                if leaves_parallel:
                    leaf_ranks.append(str(nid))
                return

            nid = _get_node_id()
            lab = _format_label(node_adapt, depth)

            # Determine fill color
            color_attr = ""
            if filled:
                fill_color = _get_fill_color(node_adapt)
                color_attr = f', fillcolor="{fill_color}"'

            # Write node
            out_file.write(f'{nid} [label="{lab}"{color_attr}] ;\n')

            # Write edge from parent
            if parent_id is not None:
                out_file.write(f"{parent_id} -> {nid}")
                if edge_label:
                    out_file.write(f' [label="{edge_label}"]')
                out_file.write(" ;\n")

            # Recurse to children
            if node_adapt.is_leaf:
                if leaves_parallel:
                    leaf_ranks.append(str(nid))
            else:
                # Handle binary children
                if node_adapt.left:
                    left_label = "\u2264" if special_characters else "True"
                    _recurse_dot(node_adapt.left, nid, left_label, depth + 1)
                if node_adapt.right:
                    right_label = ">" if special_characters else "False"
                    _recurse_dot(node_adapt.right, nid, right_label, depth + 1)

                # Handle multi-way (nominal) splits
                if node_adapt.children:
                    for child_label, child in node_adapt.children.items():
                        _recurse_dot(child, nid, str(child_label), depth + 1)

        # Generate the tree
        _recurse_dot(adapter)

        # Write leaf ranks if parallel leaves requested
        if leaves_parallel and leaf_ranks:
            out_file.write("{rank=same ; " + "; ".join(leaf_ranks) + "} ;\n")

        out_file.write("}\n")

        if return_string:
            return out_file.getvalue()
        return None

    finally:
        if own_file:
            out_file.close()


def _get_tree_root(decision_tree):
    """Extract the tree root from a model or return the node itself.

    Parameters
    ----------
    decision_tree : fitted model or tree node
        Either a fitted tree model with ``tree_`` attribute, or a tree node.

    Returns
    -------
    root : object or None
        The tree root node, or None if not found.
    """
    # If it's already a tree-like object with is_leaf attribute
    if hasattr(decision_tree, 'is_leaf'):
        return decision_tree

    # If it has tree_ attribute
    if hasattr(decision_tree, 'tree_'):
        return decision_tree.tree_

    # If it has estimators_ (RandomForest) - use first tree
    if hasattr(decision_tree, 'estimators_') and len(decision_tree.estimators_) > 0:
        first_est = decision_tree.estimators_[0]
        if hasattr(first_est, 'tree_'):
            return first_est.tree_

    return None


# Backwards compatibility alias
export_tree = export_graphviz
