"""Render fitted TuiML decision trees as figures, ASCII text, or GraphViz DOT.

This module turns the *node objects* built by TuiML's tree learners into
something a human can read. It accepts any fitted estimator from
:mod:`tuiml.algorithms.trees` -- :class:`~tuiml.algorithms.trees.DecisionTreeClassifier`
and :class:`~tuiml.algorithms.trees.DecisionTreeRegressor`,
:class:`~tuiml.algorithms.trees.C45TreeClassifier`,
:class:`~tuiml.algorithms.trees.RandomTreeClassifier`,
:class:`~tuiml.algorithms.trees.ReducedErrorPruningTreeClassifier`,
:class:`~tuiml.algorithms.trees.HoeffdingTreeClassifier`,
:class:`~tuiml.algorithms.trees.M5ModelTreeRegressor`,
:class:`~tuiml.algorithms.trees.LogisticModelTreeClassifier`, the
:class:`~tuiml.algorithms.trees.RandomForestClassifier` /
:class:`~tuiml.algorithms.trees.RandomForestRegressor` ensembles (one member
tree at a time) and :class:`~tuiml.algorithms.trees.DecisionStumpClassifier` --
as well as a bare tree-node object.

Three output formats
--------------------
:func:`plot_tree`
    A matplotlib figure of rounded/square node boxes joined by edges.
:func:`export_text`
    A plain ASCII rule listing, for terminals, logs and docstrings.
:func:`export_graphviz`
    A DOT-language string (or file) for rendering with GraphViz.

How the drawing pipeline fits together
--------------------------------------
Those learners do not share a node class: CART uses ``TreeNode``, the streaming
learner uses ``HoeffdingNode``, M5 names its split fields ``split_attr`` /
``split_value``, and the stump keeps no nodes at all. Every entry point
therefore funnels through one duck-typing shim, :class:`_TreeNodeAdapter`,
before any layout happens::

    fitted estimator
          |  _get_tree_root / _check_is_fitted   (unwrap .tree_ or .estimators_)
          v
    _TreeNodeAdapter                             (one interface for every node type)
          |  _build_bigtree_nodes                (+ _build_node_label per node)
          v
    bigtree.Node tree
          |  bigtree.reingold_tilford            (assigns tidy x / y per node)
          v
    _draw_bigtree                                (boxes, edges, edge labels -> Axes)

:func:`export_text` and :func:`export_graphviz` share the adapter but skip the
bigtree layout entirely -- they walk the adapter tree recursively and emit
characters instead of artists.

Notes
-----
Both ``matplotlib`` and ``bigtree`` are imported lazily at module load and are
optional: their absence is recorded in the module flags ``HAS_MATPLOTLIB`` and
``HAS_BIGTREE`` rather than raised. Only :func:`plot_tree` needs them, and it
raises :exc:`ImportError` at call time if either is missing. The text and DOT
exports depend on neither, so they work in a headless install.

Examples
--------
>>> from tuiml.datasets import load_iris
>>> from tuiml.algorithms.trees import DecisionTreeClassifier
>>> from tuiml.evaluation.visualization import plot_tree
>>> X, y = load_iris()
>>> clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
>>> fig = plot_tree(clf, feature_names=['sl', 'sw', 'pl', 'pw'])  # doctest: +SKIP
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
    """Generate ``n`` visually distinct colors with equally spaced hues.

    Walks the HSV hue circle at fixed saturation :math:`s = 0.75` and value
    :math:`v = 0.9`, converting each hue to RGB. Hues start at 25 and advance
    by :math:`360 / n` degrees, so no two colors sit closer than that on the
    wheel. The conversion uses the standard chroma form

    .. math::
        c = s \\cdot v, \\quad
        x = c \\left(1 - \\left|(h / 60) \\bmod 2 - 1\\right|\\right), \\quad
        m = v - c

    where the RGB triple is picked from the sextant :math:`\\lfloor h/60 \\rfloor`
    and then shifted up by :math:`m` and scaled to the 0-255 range.

    Intended for per-class node tinting (the scheme scikit-learn uses for
    ``filled=True``). The current drawing code paints nodes from the fixed
    ``_COLORS`` map instead, so this helper is retained as a utility for
    class-count-dependent palettes.

    Parameters
    ----------
    n : int
        The number of colors required. Must be at least 1.

    Returns
    -------
    color_list : list of list of int, length n
        List of ``[R, G, B]`` triples, each component an integer in 0-255.
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
    """Unified duck-typing wrapper around a node from any TuiML tree learner.

    TuiML's tree algorithms do not share a node class -- CART builds
    ``TreeNode``, the streaming learner builds ``HoeffdingNode``, M5 stores its
    split as ``split_attr`` / ``split_value`` rather than
    ``feature_index`` / ``threshold``, and C4.5 can branch many ways from a
    nominal attribute instead of two ways from a numeric one. This adapter
    hides those differences behind one read-only interface so that
    :func:`_build_bigtree_nodes`, :func:`_count_leaves`, :func:`_tree_depth`,
    :func:`export_text` and :func:`export_graphviz` can each be written once.

    Every property is computed by *probing attributes* of the wrapped node,
    never by checking its class, so a node type this module has never seen
    still renders as long as it exposes the usual field names. Child accessors
    return freshly wrapped adapters, so a whole tree is adapted lazily by
    walking from the root.

    Parameters
    ----------
    node : object
        A tree node from any TuiML tree algorithm. Recognised (all optional)
        attributes are ``is_leaf``, ``feature_index`` / ``split_attr``,
        ``threshold`` / ``split_value``, ``left``, ``right``, ``children``,
        ``is_numeric``, ``n_samples``, and the prediction fields
        ``predicted_class``, ``class_counts``, ``class_distribution``,
        ``predicted_value``, ``prediction``, ``linear_model``,
        ``weights`` / ``bias``.

    Attributes
    ----------
    _node : object
        The wrapped node, kept private so callers go through the properties.

    See Also
    --------
    :func:`~tuiml.evaluation.visualization.trees._get_tree_root` : Finds the root node to wrap.

    Examples
    --------
    >>> from tuiml.datasets import load_iris
    >>> from tuiml.algorithms.trees import DecisionTreeClassifier
    >>> from tuiml.evaluation.visualization.trees import _TreeNodeAdapter
    >>> X, y = load_iris()
    >>> clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    >>> root = _TreeNodeAdapter(clf.tree_)
    >>> bool(root.is_leaf)
    False
    >>> bool(root.left.is_leaf)
    True
    """

    def __init__(self, node):
        """Wrap a single tree node; see the class docstring for parameters."""
        self._node = node

    @property
    def is_leaf(self):
        """bool : Whether this node makes a prediction rather than a split.

        Falls back to ``False`` for nodes that do not carry the flag, so an
        unrecognised node is treated as internal and traversal continues.
        """
        return getattr(self._node, 'is_leaf', False)

    @property
    def feature_index(self):
        """int or None : Column index of the feature this node splits on.

        Prefers M5's ``split_attr`` name, then the common ``feature_index``.
        ``None`` for leaves, or for internal nodes whose split cannot be
        determined, in which case callers fall back to a placeholder label.
        """
        # M5 uses split_attr
        if hasattr(self._node, 'split_attr'):
            return self._node.split_attr
        return getattr(self._node, 'feature_index', None)

    @property
    def threshold(self):
        """float, object or None : Cut point tested at this node.

        Prefers M5's ``split_value`` name, then the common ``threshold``.
        Numeric splits send samples with :math:`x \\leq t` to
        :attr:`left`; for a nominal split the value is the category being
        matched. ``None`` when the node stores no cut point.
        """
        # M5 uses split_value
        if hasattr(self._node, 'split_value'):
            return self._node.split_value
        return getattr(self._node, 'threshold', None)

    @property
    def left(self):
        """_TreeNodeAdapter or None : Wrapped "condition true" child.

        Returns a new adapter so traversal stays uniform, or ``None`` when the
        node has no left child (leaves, and multi-way nominal splits which use
        :attr:`children` instead).
        """
        child = getattr(self._node, 'left', None)
        return _TreeNodeAdapter(child) if child is not None else None

    @property
    def right(self):
        """_TreeNodeAdapter or None : Wrapped "condition false" child.

        Returns a new adapter so traversal stays uniform, or ``None`` when the
        node has no right child.
        """
        child = getattr(self._node, 'right', None)
        return _TreeNodeAdapter(child) if child is not None else None

    @property
    def children(self):
        """dict or None : Wrapped branches of a multi-way (nominal) split.

        C4.5-style learners branch once per category of a nominal attribute
        rather than twice on a numeric threshold, storing the branches in a
        ``{category: node}`` mapping. Each value is wrapped in its own adapter
        and the key becomes the edge label when drawn.

        Returns
        -------
        children : dict of {object : _TreeNodeAdapter} or None
            ``None`` when the node has no such mapping (the binary case).
        """
        ch = getattr(self._node, 'children', None)
        if ch and isinstance(ch, dict):
            return {k: _TreeNodeAdapter(v) for k, v in ch.items()}
        return None

    @property
    def is_nominal_split(self):
        """bool : Whether this node branches many ways instead of two.

        True when the node explicitly declares ``is_numeric = False``, or when
        it is an internal node carrying a non-empty ``children`` mapping.
        Traversal code uses this to choose between the
        :attr:`left`/:attr:`right` pair and the :attr:`children` mapping.
        """
        node = self._node
        if getattr(node, 'is_numeric', True) is False:
            return True
        if hasattr(node, 'children') and isinstance(getattr(node, 'children', None), dict):
            if not self.is_leaf and getattr(node, 'children', None):
                return True
        return False

    @property
    def n_samples(self):
        """int or None : Training samples that reached this node.

        Shown as the ``samples = N`` line of a node label. ``None`` when the
        learner does not record it, in which case the line is omitted.
        """
        return getattr(self._node, 'n_samples', None)

    @property
    def is_classification(self):
        """bool : Whether this node predicts a class rather than a number.

        Decided by the presence of any classification-flavoured payload:
        ``predicted_class``, ``class_distribution``, ``class_counts``, or a
        ``weights``/``bias`` pair (the logistic model held at a Logistic Model
        Tree leaf). Drives both the label wording (``class =`` versus
        ``value =``) and the leaf fill color.
        """
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
        """Return a one-line, human-readable split condition for this node.

        Produces strings of the form ``"petal_width <= 0.8"``. Floats are
        formatted with ``%.3g`` so a threshold never widens a node box with
        noise digits. When no name is supplied for the split feature, a
        positional placeholder ``X[i]`` is used instead.

        Parameters
        ----------
        feature_names : list of str, optional
            Names indexed by feature column. Entries beyond the list length
            fall back to the ``X[i]`` placeholder.

        Returns
        -------
        label : str
            The condition, the bare feature name when the node has no
            threshold, or ``'?'`` when the split feature is unknown.

        See Also
        --------
        :func:`~tuiml.evaluation.visualization.trees._build_node_label` : Builds the full multi-line box label.
        """
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
        """Return a human-readable prediction string for a leaf node.

        Every tree learner stores its leaf payload differently, so this walks
        a fixed preference order and formats whichever field it finds first::

            predicted_class   -> the class, mapped through class_names
            class_counts      -> majority class of a streaming leaf's counters
            predicted_value   -> regression value, %.3g
            prediction        -> regression value, %.3g
            linear_model      -> the literal "LM" (M5 model-tree leaf)

        A ``n=<samples>`` line is appended on a second line when the node
        records a sample count. Callers that need a single line (the box
        labels, the text export) split on the newline and keep the first part.

        Parameters
        ----------
        class_names : list of str, optional
            Class names in ascending numerical order. A class value that
            cannot be used as an index into this list is left as-is rather
            than raising.

        Returns
        -------
        label : str
            The formatted prediction, possibly with a trailing ``"\\nn=..."``
            line, or ``'?'`` when no recognised payload is present.
        """
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
        """Return the hex fill color that encodes this node's role.

        Three-way scheme drawn from the module's ``_COLORS`` map: internal
        (splitting) nodes are steel blue, classification leaves sage green,
        and regression leaves coral. Used only when ``filled=True``;
        :func:`_get_node_facecolor` overrides it with gray for nodes hidden
        behind a ``max_depth`` truncation.

        Returns
        -------
        color : str
            A ``'#RRGGBB'`` hex color string.
        """
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
    """Build the multi-line text that goes inside one node box.

    Called once per node by :func:`_build_bigtree_nodes`, and the resulting
    string is what :func:`_compute_box_size` measures to size the box. Lines
    are assembled top to bottom and joined with newlines::

        petal_width <= 0.8      <- split condition (internal nodes only)
        samples = 150           <- only when label_mode permits
        class = setosa          <- leaves only ("value = ..." if regression)

    Parameters
    ----------
    adapter : _TreeNodeAdapter
        The node to label.
    feature_names : list of str, optional
        Names indexed by feature column. Missing names become ``X[i]``.
    class_names : list of str, optional
        Class names in ascending numerical order, used for leaf predictions.
    label_mode : {'all', 'root', 'none'}, default='all'
        Which nodes get the informative ``samples = ...`` line: every node,
        the root only, or none. The split condition and the leaf prediction
        are always shown.
    precision : int, default=3
        Significant digits (``%g``) for a floating-point threshold.
    depth : int, default=0
        Depth of this node, needed to resolve ``label_mode='root'``.

    Returns
    -------
    label : str
        Newline-joined label, or ``'?'`` when nothing could be derived.
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
    """Recursively mirror an adapter tree into a ``bigtree`` node tree.

    This is the bridge between TuiML's node objects and the layout engine:
    :func:`bigtree.reingold_tilford` can only assign coordinates to its own
    ``Node`` type, so every adapter node is copied into a ``BTNode`` whose
    label text and drawing flags are baked in up front. After this call,
    nothing downstream needs to look at a TuiML node again -- except through
    the ``node_adapter`` attribute kept for color lookup.

    Each emitted ``BTNode`` carries the custom attributes ``node_label`` (the
    text from :func:`_build_node_label`), ``is_leaf_node``, ``is_truncated``,
    ``is_classification``, ``node_adapter`` and ``edge_label``. Node names are
    stringified counter values, unique across the tree, because bigtree
    requires sibling names to differ.

    Branching mirrors the source: a nominal split emits one child per category
    (edge label = the category), a numeric split emits left then right (edge
    labels ``"True"`` and ``"False"``). Recursion stops at leaves and at
    ``depth >= max_depth``, where a placeholder ``(...)`` node marks the
    elision.

    Parameters
    ----------
    adapter : _TreeNodeAdapter
        Adapter for the node to convert; the root on the first call.
    feature_names : list of str, optional
        Feature names forwarded to :func:`_build_node_label`.
    class_names : list of str, optional
        Class names forwarded to :func:`_build_node_label`.
    max_depth : int, optional
        Deepest level to expand. Nodes at that depth that would otherwise
        split are replaced by a truncation marker. ``None`` expands fully.
    depth : int, default=0
        Depth of the current node; incremented through the recursion.
    counter : list of int, optional
        Single-element list used as a mutable counter to hand out unique node
        names across the whole recursion. Created on the first call.
    label_mode : {'all', 'root', 'none'}, default='all'
        Forwarded to :func:`_build_node_label`.
    precision : int, default=3
        Forwarded to :func:`_build_node_label`.

    Returns
    -------
    bt_node : bigtree.Node
        Root of the converted subtree, with children already attached.

    See Also
    --------
    :func:`~tuiml.evaluation.visualization.trees._draw_bigtree` : Consumes the laid-out result.
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
    """Yield every node of a bigtree subtree in pre-order (parents first).

    :func:`_draw_bigtree` needs the node list several times -- to size boxes,
    to draw edges, then to draw boxes -- so it materialises this generator
    once. Pre-order guarantees a parent is seen before its children.

    Parameters
    ----------
    bt_node : bigtree.Node
        Root of the subtree to walk.

    Yields
    ------
    node : bigtree.Node
        Each node in the subtree, starting with ``bt_node`` itself.
    """
    yield bt_node
    for child in bt_node.children:
        yield from _iter_nodes(child)


def _get_node_facecolor(node, filled):
    """Resolve the final box fill color for one laid-out node.

    Applies the two overrides that sit on top of the adapter's own scheme:
    an unfilled plot is always white, and a ``max_depth`` truncation marker is
    always gray regardless of what it stands in for. Anything else defers to
    :meth:`_TreeNodeAdapter.get_node_color`.

    Parameters
    ----------
    node : bigtree.Node
        A node produced by :func:`_build_bigtree_nodes`, carrying the
        ``is_truncated`` and ``node_adapter`` attributes.
    filled : bool
        Whether the caller asked for colored boxes.

    Returns
    -------
    color : str
        ``'white'`` or a ``'#RRGGBB'`` hex string.
    """
    if not filled:
        return 'white'
    if node.is_truncated:
        return _COLORS['truncated']
    return node.node_adapter.get_node_color()


def _compute_box_size(label, fontsize):
    """Estimate the data-space width and height of one node box.

    matplotlib cannot report text extents before a draw pass, so the box is
    sized analytically from the label's longest line and line count. With
    :math:`L` the longest line length in characters, :math:`N` the number of
    lines and :math:`f` the font size in points

    .. math::
        w = \\max(1.2,\\; 0.009 \\, f L + 0.4), \\qquad
        h = \\max(0.4,\\; 0.032 \\, f N + 0.25)

    The constants are per-character advance and per-line height calibrated for
    the sans-serif face; the additive terms are padding and the floors keep a
    one-word box from collapsing. :func:`_draw_bigtree` takes the maximum
    width and height over all nodes and uses them to scale the layout's unit
    spacing, which is what actually prevents overlap.

    Parameters
    ----------
    label : str
        Node text, possibly containing embedded newlines.
    fontsize : int or float
        Font size in points that the label will be drawn at.

    Returns
    -------
    box_w : float
        Box width in data units.
    box_h : float
        Box height in data units.
    """
    lines = label.split('\n')
    max_line_len = max((len(l) for l in lines), default=1)
    n_lines = len(lines)
    char_w = fontsize * 0.009
    line_h = fontsize * 0.032
    box_w = max(1.2, max_line_len * char_w + 0.4)
    box_h = max(0.4, n_lines * line_h + 0.25)
    return box_w, box_h


def _draw_bigtree(bt_root, ax, filled=False, rounded=True, fontsize=None):
    """Render a laid-out bigtree onto matplotlib axes as boxes and edges.

    The final stage of the drawing pipeline. It expects
    :func:`bigtree.reingold_tilford` to have already stamped an ``x`` and ``y``
    on every node; those coordinates are in *layout units* where sibling
    spacing is 1.0, which has nothing to do with how wide the text is. This
    function's real job is turning those units into a non-overlapping picture:

    1. Pick a font size from the node count when none was given, shrinking by
       one point per five nodes within the range 8-12.
    2. Measure every box with :func:`_compute_box_size` and take the maxima
       :math:`w_{max}`, :math:`h_{max}`.
    3. Rescale every coordinate in place so one layout unit is wider than the
       widest box, :math:`x \\mapsto 1.3 \\, w_{max} x` and
       :math:`y \\mapsto 1.8 \\, h_{max} y`. The multipliers are the gap.
    4. Draw edges first (``zorder=1``), then boxes (3) and text (4), so no
       line crosses a label.

    Edges leave a parent's bottom edge and land on a child's top edge, and
    only the root's own children get their ``True`` / ``False`` labels drawn,
    placed at the midpoint and pushed away from the trunk::

              +------------------+
              | petal_w <= 0.8   |     y = parent.y
              |  samples = 150   |
              +--------+---------+
                 /            \\        <- edge from box bottom to box top
          True  /              \\  False
               /                \\
        +-----------+      +-----------+
        | class = 0 |      | ...       |   y = parent.y - 1.8 * h_max
        +-----------+      +-----------+
        x = parent.x - dx   x = parent.x + dx

    Axis limits are finally padded by one box width and height so nothing is
    clipped at the border.

    Parameters
    ----------
    bt_root : bigtree.Node
        Root node. Must already carry ``x`` / ``y`` from
        :func:`bigtree.reingold_tilford`, plus the attributes attached by
        :func:`_build_bigtree_nodes`.
    ax : matplotlib.axes.Axes
        Target axes. Patches and text are added to it and its limits are set.
    filled : bool, default=False
        Fill boxes with the node's semantic color; text switches to white on
        dark fills for contrast.
    rounded : bool, default=True
        Use rounded rather than square box corners.
    fontsize : int or None, default=None
        Point size for node text. Auto-computed from the node count if None.

    Returns
    -------
    None
        Draws onto ``ax`` in place; returns early if the tree is empty.

    Notes
    -----
    Mutates ``node.x`` and ``node.y`` on the bigtree nodes it is given. The
    coordinates are therefore only valid for a single draw -- callers build a
    fresh bigtree per plot rather than re-drawing one.
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
    """Count the boxes that will appear on the bottom fringe of the drawing.

    Called by :func:`plot_tree` to pick a default figure *width*, since the
    horizontal extent of a Reingold-Tilford layout is driven by the number of
    terminal nodes rather than by the total node count. A node elided by
    ``max_depth`` counts as one leaf because it is still drawn, as a single
    ``(...)`` marker.

    Parameters
    ----------
    adapter : _TreeNodeAdapter
        Subtree root to count under.
    max_depth : int, optional
        Depth at which counting stops and the subtree collapses to one
        truncation marker. ``None`` counts the whole tree.
    depth : int, default=0
        Depth of the current node; incremented through the recursion.

    Returns
    -------
    n_leaves : int
        Number of leaves plus truncation markers, always at least 1.
    """
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
    """Compute how many levels of boxes the drawing will stack.

    The vertical counterpart of :func:`_count_leaves`: :func:`plot_tree` uses
    it to pick a default figure *height*. Depth is the longest root-to-leaf
    path measured in edges, so a single-node tree has depth 0. Capped at
    ``max_depth`` when truncation is in effect, matching what is drawn.

    Parameters
    ----------
    adapter : _TreeNodeAdapter
        Subtree root to measure under.
    max_depth : int, optional
        Depth at which measuring stops. ``None`` measures the whole tree.
    depth : int, default=0
        Depth of the current node; incremented through the recursion.

    Returns
    -------
    depth : int
        Number of edges on the longest path below (and including) this node.
    """
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
    """Report whether a tree estimator has been fitted, without calling it.

    The guard :func:`plot_tree` runs before it starts unwrapping a model, so
    an unfitted estimator produces a clear ``ValueError`` rather than an
    ``AttributeError`` from somewhere inside the layout code. Fitted state is
    inferred from whichever marker the estimator family sets, in order:

    ================================ ==========================================
    Marker attribute                 Estimator family
    ================================ ==========================================
    ``tree_`` is not None            single trees (CART, C4.5, REP, M5, LMT)
    ``estimators_`` is non-empty     forests and other ensembles
    ``feature_index_`` is not None   decision stumps, which build no nodes
    ``n_features_in_`` present       scikit-learn-style fitted convention
    ================================ ==========================================

    Parameters
    ----------
    model : object
        The estimator to test. Bare tree nodes are not recognised here.

    Returns
    -------
    is_fitted : bool
        True when any marker above is satisfied, False otherwise.
    """
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
    """Draw a fitted decision tree as a matplotlib figure of labelled boxes.

    Each internal node is drawn as a box holding its split condition, each
    leaf as a box holding its prediction, and edges run from a parent's bottom
    edge to a child's top edge. Positions come from the Reingold-Tilford tidy
    tree algorithm, which centres each parent over its children and keeps
    subtrees from overlapping; box sizes are then measured from the label text
    and the layout is scaled so no two boxes collide.

    The signature follows scikit-learn's ``plot_tree`` so existing calls port
    over, with four TuiML-only additions (``tree_index``, ``figsize``,
    ``save_path``, ``title``) at the end.

    Parameters
    ----------
    decision_tree : object
        The fitted tree to plot. Accepts a single-tree estimator with a
        ``tree_`` attribute (for example
        :class:`~tuiml.algorithms.trees.DecisionTreeClassifier`,
        :class:`~tuiml.algorithms.trees.C45TreeClassifier`,
        :class:`~tuiml.algorithms.trees.M5ModelTreeRegressor`), an ensemble
        with ``estimators_`` such as
        :class:`~tuiml.algorithms.trees.RandomForestClassifier` (see
        ``tree_index``), a fitted
        :class:`~tuiml.algorithms.trees.DecisionStumpClassifier`, or a bare
        tree node object.

    max_depth : int, default=None
        The maximum depth of the representation. If None, the tree is fully
        generated.

    feature_names : array-like of str, default=None
        Names of each of the features.
        If None, generic names will be used ("X[0]", "X[1]", ...).

    class_names : array-like of str, default=None
        Names of each of the target classes in ascending numerical order, used
        to label leaves. Only relevant for classification. A class value that
        cannot index this list is printed as-is rather than raising.

    label : {'all', 'root', 'none'}, default='all'
        Which nodes get the informative ``samples = ...`` line: 'all' shows it
        at every node, 'root' only at the top node, 'none' at no node. Split
        conditions and leaf predictions are always drawn.

    filled : bool, default=False
        When ``True``, fill node boxes with a color encoding the node's role:
        blue for internal nodes, green for classification leaves, orange for
        regression leaves, gray for ``max_depth`` truncation markers.

    impurity : bool, default=True
        Accepted for scikit-learn signature compatibility but currently
        ignored: TuiML node labels do not include an impurity line.

    node_ids : bool, default=False
        Accepted for scikit-learn signature compatibility but currently
        ignored by this function. :func:`export_graphviz` does honour it.

    proportion : bool, default=False
        Accepted for scikit-learn signature compatibility but currently
        ignored; sample counts are always drawn as absolute counts.

    rounded : bool, default=False
        When ``True``, draw node boxes with rounded corners instead of square
        ones. Fonts are unaffected.

    precision : int, default=3
        Number of significant digits (``%g``) used when formatting a
        floating-point threshold in a node label.

    ax : matplotlib.axes.Axes, default=None
        Axes to draw on. If None, a new figure and axes are created using
        ``figsize`` and the TuiML plot style.

    fontsize : int, default=None
        Point size for node text. If None, chosen from the node count within
        the range 8-12 so that large trees stay legible.

    tree_index : int, default=0
        For ensembles with ``estimators_``, which member tree to draw
        (TuiML extension).

    figsize : tuple of float, default=None
        Figure size ``(width, height)`` in inches (TuiML extension). Ignored
        when ``ax`` is given. If None, computed from the tree's leaf count and
        depth as ``(max(10, 3 * n_leaves), max(6, 2.5 * (depth + 1)))``.

    save_path : str, default=None
        If given, the figure is also written to this path as a 300 dpi PNG
        (TuiML extension).

    title : str, default=None
        Custom plot title (TuiML extension). Defaults to the estimator's class
        name, or ``"Random Forest - Tree <i>"`` for an ensemble member.

    Returns
    -------
    annotations : list
        Always an empty list. The return value exists for signature
        compatibility with scikit-learn's ``plot_tree``, which returns the
        annotation artists; this implementation draws
        :class:`~matplotlib.patches.FancyBboxPatch` boxes directly onto the
        axes instead and does not collect them. Retrieve the figure with
        ``ax.get_figure()`` or ``plt.gcf()`` if you need it.

    Raises
    ------
    ImportError
        If ``matplotlib`` or ``bigtree`` is not installed. Both are optional
        dependencies imported lazily at module load; install the latter with
        ``pip install bigtree``.
    ValueError
        If ``decision_tree`` is not fitted, or if ``tree_index`` is out of
        range for the ensemble's ``estimators_``.

    See Also
    --------
    :func:`~tuiml.evaluation.visualization.trees.export_text` : Same tree as an ASCII rule listing.
    :func:`~tuiml.evaluation.visualization.trees.export_graphviz` : Same tree as GraphViz DOT.

    Notes
    -----
    The figure is displayed with ``plt.show()`` before returning, so in a
    script with an interactive backend this call blocks until the window is
    closed. Set a non-interactive backend (``matplotlib.use('Agg')``) when
    plotting headlessly, and pass ``save_path`` to keep the result.

    Examples
    --------
    >>> from tuiml.datasets import load_iris
    >>> from tuiml.algorithms.trees import DecisionTreeClassifier
    >>> from tuiml.evaluation.visualization import plot_tree
    >>> X, y = load_iris()
    >>> names = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w']
    >>> clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    >>> plot_tree(clf, feature_names=names, filled=True)      # doctest: +SKIP

    Draw only the top two levels of one tree out of a fitted forest, and save
    it instead of inspecting it interactively:

    >>> from tuiml.algorithms.trees import RandomForestClassifier
    >>> rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    >>> plot_tree(rf, tree_index=3, max_depth=2, rounded=True,
    ...           save_path='tree3.png')                      # doctest: +SKIP
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
    """Draw a fitted DecisionStump, which stores no node objects to adapt.

    :class:`~tuiml.algorithms.trees.DecisionStumpClassifier` keeps its whole
    model in flat fitted attributes (``feature_index_``, ``threshold_``,
    ``left_class_``, ``right_class_``) rather than in a linked node structure,
    so :class:`_TreeNodeAdapter` has nothing to wrap. :func:`plot_tree`
    delegates here whenever it sees ``feature_index_`` without ``tree_``.

    Rather than draw by hand, this builds the three node labels itself,
    synthesises a 3-node bigtree (root split, left leaf, right leaf) with a
    minimal private ``_StumpAdapter`` standing in for the node adapter, and
    then reuses :func:`bigtree.reingold_tilford` and :func:`_draw_bigtree`.
    The result is visually identical to any other tree plot.

    Parameters
    ----------
    model : DecisionStumpClassifier
        A fitted stump.
    feature_names : list of str or None
        Names indexed by feature column; ``X[i]`` is used when absent.
    class_names : list of str or None
        Class names used to label the two leaves.
    figsize : tuple of float or None
        Figure size in inches; defaults to ``(8, 5)``, which fits three boxes.
    save_path : str or None
        If given, the figure is written there as a 300 dpi PNG.
    title : str or None
        Plot title; defaults to ``'Decision Stump'``.
    fontsize : int or None
        Point size for node text; auto-computed when None.
    filled : bool
        Fill node boxes with their role color.
    rounded : bool
        Use rounded box corners.

    Returns
    -------
    annotations : list
        Always empty, matching :func:`plot_tree`'s return contract.

    Notes
    -----
    Calls ``plt.show()`` before returning, exactly like :func:`plot_tree`.
    """
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
        """Minimal stand-in for :class:`_TreeNodeAdapter`, for a stump's 3 synthesized nodes.

        :class:`~tuiml.algorithms.trees.DecisionStumpClassifier` keeps no
        linked node objects, so there is nothing for :class:`_TreeNodeAdapter`
        to wrap when :func:`_plot_decision_stump_legacy` builds its root and
        two leaf ``BTNode``\\ s by hand. Each of those three nodes gets one of
        these instead, carrying just enough state -- ``is_leaf`` and
        ``is_classification`` -- for :func:`_get_node_facecolor` to call
        :meth:`get_node_color` on it exactly as it would on a real adapter.

        Parameters
        ----------
        is_leaf : bool
            Whether this stand-in represents a leaf (``True``) or the root
            split (``False``).
        is_cls : bool
            Whether the stump is a classifier. Always ``True`` at the call
            sites in this module, since :class:`_StumpAdapter` only backs
            :class:`~tuiml.algorithms.trees.DecisionStumpClassifier`.
        """

        def __init__(self, is_leaf, is_cls):
            """Store the leaf/classification flags used by :meth:`get_node_color`."""
            self.is_leaf = is_leaf
            self.is_classification = is_cls

        def get_node_color(self):
            """Return the hex fill color for this stand-in node.

            Mirrors :meth:`_TreeNodeAdapter.get_node_color`, but reads the
            ``is_leaf`` / ``is_classification`` flags stored on ``self``
            directly instead of probing a wrapped node.

            Returns
            -------
            color : str
                ``_COLORS['internal']`` for the root, ``_COLORS['leaf_cls']``
                for a classification leaf, or ``_COLORS['leaf_reg']`` for a
                regression leaf.
            """
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
    >>> from tuiml.algorithms.trees import DecisionTreeClassifier
    >>> from tuiml.datasets import load_iris
    >>> X, y = load_iris()
    >>> from tuiml.evaluation.visualization.trees import export_text
    >>> clf = DecisionTreeClassifier().fit(X, y)
    >>> print(export_text(clf, feature_names=['a', 'b']))   # doctest: +SKIP
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
    >>> from tuiml.algorithms.trees import DecisionTreeClassifier
    >>> from tuiml.datasets import load_iris
    >>> X, y = load_iris()
    >>> from tuiml.evaluation.visualization.trees import export_graphviz
    >>> clf = DecisionTreeClassifier().fit(X, y)
    >>> dot_data = export_graphviz(clf, feature_names=['a', 'b'])
    >>> print(dot_data)                                    # doctest: +SKIP
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
            """Hand out the next sequential integer DOT node name.

            Every call advances the shared ``node_counter`` closure variable, so
            each node written by :func:`_recurse_dot` -- including truncation
            markers -- gets a distinct, increasing id, matching the order nodes
            are visited (pre-order, parent before children).

            Returns
            -------
            nid : int
                The node id to use for the current node, starting at 0.
            """
            nid = node_counter[0]
            node_counter[0] += 1
            return nid

        def _format_label(node_adapt, depth, node_id):
            """Format the label for a node.

            ``node_id`` is passed in rather than read from ``node_counter``:
            the caller has already claimed its id via ``_get_node_id()``, which
            advanced the counter, so reading it here would label every node
            with the id of the node that comes next.
            """
            lines = []

            # Node ID if requested
            if node_ids:
                lines.append(f"node #{node_id}")

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
            lab = _format_label(node_adapt, depth, nid)

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
