"""
Matplotlib plots for reading, comparing and reporting evaluation results.

This package turns the numbers produced by :mod:`tuiml.evaluation.metrics` and
:mod:`tuiml.evaluation.statistics` into publication-quality figures. Reach for
it at the *end* of an experiment, when you already have predictions or a score
matrix and want to see — or publish — what they say.

Two families of plots live here:

**Single-model diagnostics** — you have one fitted model and its predictions on
a test set:

- :func:`~tuiml.evaluation.visualization.plot_confusion_matrix` — which classes
  get mistaken for which.
- :func:`~tuiml.evaluation.visualization.plot_roc_curve` /
  :func:`~tuiml.evaluation.visualization.plot_pr_curve` — threshold-free ranking
  quality; the PR curve is the one to trust on imbalanced data.
- :func:`~tuiml.evaluation.visualization.plot_learning_curve` — whether more
  training data would still help.
- :func:`~tuiml.evaluation.visualization.plot_tree` — the structure of a fitted
  decision tree.

**Multi-model comparison** — you have a ``(n_datasets, n_algorithms)`` score
matrix from a benchmark:

- :func:`~tuiml.evaluation.visualization.plot_critical_difference` — the
  Demšar (2006) rank diagram: which algorithms are *statistically*
  distinguishable, not merely different on average.
- :func:`~tuiml.evaluation.visualization.plot_ranking_table` — the raw numbers
  plus per-dataset ranks.
- :func:`~tuiml.evaluation.visualization.plot_boxplot_comparison` — score spread
  per algorithm.
- :func:`~tuiml.evaluation.visualization.plot_heatmap` — the whole score matrix
  at a glance.

All plotting functions share the same conventions: matplotlib is imported
lazily, so they raise :exc:`ImportError` if it is unavailable; each one calls
``matplotlib.pyplot.show()`` before returning, and writes a 300-dpi PNG when a
``save_path`` is supplied. Styling comes from a shared internal stylesheet whose
palettes (``PALETTES``) and semantic colour tokens (``SEMANTIC_COLORS``) are
re-exported here, together with the helpers ``apply_style``, ``reset_style``,
``get_colors``, ``setup_figure`` and ``style_axis`` for building your own
matching figures.

Notes
-----
``seaborn`` is an optional accelerant: when installed,
:func:`~tuiml.evaluation.visualization.plot_confusion_matrix` and
:func:`~tuiml.evaluation.visualization.plot_heatmap` render through
``seaborn.heatmap``; otherwise they fall back to plain matplotlib and look
slightly different. Nothing here requires it.

Examples
--------
>>> import numpy as np
>>> from tuiml.evaluation.visualization import plot_confusion_matrix
>>> y_true = np.array([0, 1, 1, 0, 2, 2, 1])
>>> y_pred = np.array([0, 1, 0, 0, 2, 1, 1])
>>> cm = plot_confusion_matrix(y_true, y_pred)   # doctest: +SKIP
"""

from .cd_diagram import (
    plot_critical_difference,
    compute_ranks,
    critical_difference,
    CDDiagramResult,
)
from .comparison import (
    plot_ranking_table,
    plot_boxplot_comparison,
    plot_heatmap,
)
from .curves import (
    plot_roc_curve,
    plot_pr_curve,
    plot_learning_curve,
)
from .confusion import (
    plot_confusion_matrix,
)
from .trees import (
    plot_tree,
)
from ._style import (
    apply_style,
    reset_style,
    get_colors,
    setup_figure,
    style_axis,
    PALETTES,
    SEMANTIC_COLORS,
)

__all__ = [
    # CD Diagram
    "plot_critical_difference",
    "compute_ranks",
    "critical_difference",
    "CDDiagramResult",
    # Comparison
    "plot_ranking_table",
    "plot_boxplot_comparison",
    "plot_heatmap",
    # Curves
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_learning_curve",
    # Confusion
    "plot_confusion_matrix",
    # Trees
    "plot_tree",
    # Styling
    "apply_style",
    "reset_style",
    "get_colors",
    "setup_figure",
    "style_axis",
    "PALETTES",
    "SEMANTIC_COLORS",
]
