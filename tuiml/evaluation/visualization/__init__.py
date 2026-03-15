"""
Visualization tools for model evaluation results.

Includes:
- Critical Difference diagrams
- Confusion matrix plots
- ROC/PR curves
- Boxplot comparisons
- Ranking tables
- Modern styling utilities
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
