"""
Confusion matrix visualization.
"""

import numpy as np
from typing import List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from ._style import apply_style, setup_figure, style_axis, SEMANTIC_COLORS

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (8, 6),
    save_path: str = None,
    normalize: bool = False,
    cmap: str = 'Blues',
    show_values: bool = True,
    show_colorbar: bool = True,
    show_grid: bool = False,
):
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : ndarray
        True labels.
    y_pred : ndarray
        Predicted labels.
    labels : list of str, optional
        Class labels for display.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    normalize : bool, default=False
        Normalize the confusion matrix (show percentages).
    cmap : str, default='Blues'
        Colormap name.
    show_values : bool, default=True
        Show values in cells.
    show_colorbar : bool, default=True
        Show colorbar.
    show_grid : bool, default=False
        Whether to show axis grid lines.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    # Apply modern styling
    apply_style()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    # Create class mapping
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Compute confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        i = class_to_idx[true]
        j = class_to_idx[pred]
        cm[i, j] += 1

    # Normalize if requested
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)
    else:
        cm_display = cm

    # Labels
    if labels is None:
        labels = [str(c) for c in classes]

    # Plot
    fig, ax = setup_figure(figsize=figsize)

    if HAS_SEABORN:
        fmt = '.1%' if normalize else 'd'
        sns.heatmap(
            cm_display, annot=show_values, fmt=fmt, cmap=cmap,
            xticklabels=labels, yticklabels=labels, square=True,
            cbar=show_colorbar, ax=ax, linewidths=1.0, linecolor='white',
            annot_kws={'fontsize': 14, 'fontweight': 'bold'},
        )
        if show_colorbar:
            cbar = ax.collections[0].colorbar
            if cbar is not None:
                cbar.set_label('Count' if not normalize else 'Proportion',
                               fontsize=13, fontweight='bold')
                cbar.ax.tick_params(labelsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12, fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12, fontweight='bold')
    else:
        im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap, aspect='equal')

        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Count' if not normalize else 'Proportion',
                           fontsize=13, fontweight='bold')
            cbar.ax.tick_params(labelsize=12)

        # Labels and ticks
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12, fontweight='bold')
        ax.set_yticklabels(labels, fontsize=12, fontweight='bold')

        # Remove spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Show values in cells
        if show_values:
            thresh = (cm_display.max() + cm_display.min()) / 2

            for i in range(n_classes):
                for j in range(n_classes):
                    if normalize:
                        text = f'{cm_display[i, j]:.1%}'
                    else:
                        text = str(cm_display[i, j])

                    color = 'white' if cm_display[i, j] > thresh else SEMANTIC_COLORS['text']
                    ax.text(j, i, text, ha='center', va='center',
                            color=color, fontsize=14, fontweight='bold')
    style_axis(
        ax,
        title=title,
        xlabel='Predicted Label',
        ylabel='True Label',
        legend=False,
        grid=show_grid,
        despine=not HAS_SEABORN,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)

    plt.show()

    return cm
