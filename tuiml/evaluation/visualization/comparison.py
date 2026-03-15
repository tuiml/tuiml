"""
Model comparison visualizations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    import pandas as pd
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from .cd_diagram import compute_ranks
from ._style import apply_style, get_colors, setup_figure, style_axis, SEMANTIC_COLORS

def plot_ranking_table(
    scores: Union[np.ndarray, Dict[str, np.ndarray]],
    names: List[str] = None,
    dataset_names: List[str] = None,
    lower_better: bool = False,
    metric_name: str = 'Score',
    figsize: Tuple[int, int] = None,
    save_path: str = None,
    show_ranks: bool = True,
    precision: int = 3,
    title: str = None,
):
    """
    Plot a ranking table with scores and ranks.

    Parameters
    ----------
    scores : ndarray of shape (n_datasets, n_algorithms) or dict
        Performance scores matrix.
    names : list of str, optional
        Algorithm names.
    dataset_names : list of str, optional
        Dataset names.
    lower_better : bool, default=False
        If True, lower scores are better.
    metric_name : str, default='Score'
        Name of the metric.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save the figure.
    show_ranks : bool, default=True
        Whether to show ranks in parentheses.
    precision : int, default=3
        Decimal precision for scores.
    title : str, optional
        Custom title. Set to empty string '' to hide title.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    # Apply modern styling
    apply_style()

    if isinstance(scores, dict):
        names = list(scores.keys())
        scores = np.array([scores[name] for name in names]).T

    scores = np.asarray(scores)
    n_datasets, n_algorithms = scores.shape

    if names is None:
        names = [f'Alg{i+1}' for i in range(n_algorithms)]

    if dataset_names is None:
        dataset_names = [f'Dataset{i+1}' for i in range(n_datasets)]

    # Compute ranks
    ranks = compute_ranks(scores, lower_better=lower_better)
    avg_ranks = np.mean(ranks, axis=0)
    sum_ranks = np.sum(ranks, axis=0)

    # Create cell text
    cell_text = []
    for i in range(n_datasets):
        row = []
        for j in range(n_algorithms):
            if show_ranks:
                row.append(f'{scores[i, j]:.{precision}f} ({int(ranks[i, j])})')
            else:
                row.append(f'{scores[i, j]:.{precision}f}')
        cell_text.append(row)

    # Add summary rows
    cell_text.append([f'{s:.1f}' for s in sum_ranks])
    cell_text.append([f'{a:.2f}' for a in avg_ranks])

    row_labels = dataset_names + ['Sum Ranks', 'Avg Rank']

    # Calculate appropriate figure size based on content
    n_rows = n_datasets + 3  # data rows + header + 2 summary rows
    max_name_len = max(len(name) for name in names)
    if figsize is None:
        col_width = max(2.0, max_name_len * 0.14)
        width = max(8, n_algorithms * col_width + 2)
        height = max(2, n_rows * 0.6 + 0.5)
        figsize = (width, height)

    fig, ax = setup_figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.axis('off')

    # Create table — no fixed bbox so auto_set_column_width can adjust
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=names,
        cellLoc='center',
        loc='upper center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.auto_set_column_width(list(range(-1, n_algorithms)))
    table.scale(1.0, 1.8)

    # Shrink axes to remove vertical gap between title and table
    ax.set_position([ax.get_position().x0, 0.0,
                     ax.get_position().width, 0.92])

    # Style header row
    for j, name in enumerate(names):
        cell = table[(0, j)]
        cell.set_facecolor(SEMANTIC_COLORS['primary'])
        cell.set_text_props(color='white', fontweight='bold')

    # Style row labels column
    for i, label in enumerate(row_labels):
        cell = table[(i + 1, -1)]
        cell.set_facecolor('#F3F4F6')
        cell.set_text_props(fontweight='medium')

    # Highlight best values with modern green
    for i in range(n_datasets):
        best_idx = np.argmin(scores[i]) if lower_better else np.argmax(scores[i])
        table[(i + 1, best_idx)].set_facecolor('#BBF7D0')

    # Highlight best average rank with modern gold
    best_avg_idx = np.argmin(avg_ranks)
    table[(n_datasets + 2, best_avg_idx)].set_facecolor('#FDE68A')

    # Style summary rows
    for j in range(n_algorithms):
        table[(n_datasets + 1, j)].set_facecolor('#E5E7EB')
        table[(n_datasets + 2, j)].set_facecolor('#E5E7EB')

    # Add borders
    for key, cell in table.get_celld().items():
        cell.set_edgecolor(SEMANTIC_COLORS['border'])
        cell.set_linewidth(1.0)

    # Title (only if not empty string)
    if title is None:
        title = f'{metric_name} Table with Rankings'
    if title:
        ax.set_title(title.title(), fontsize=16, fontweight='bold', pad=10)

    fig.subplots_adjust(top=0.92, bottom=0.02)

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)

    plt.show()

def plot_boxplot_comparison(
    scores: Union[np.ndarray, Dict[str, np.ndarray]],
    names: List[str] = None,
    metric_name: str = 'Score',
    figsize: Tuple[int, int] = (12, 6),
    save_path: str = None,
    show_mean: bool = True,
    notch: bool = False,
    show_grid: bool = False,
):
    """
    Plot boxplot comparison of algorithms.

    Parameters
    ----------
    scores : ndarray of shape (n_datasets, n_algorithms) or dict
        Performance scores matrix.
    names : list of str, optional
        Algorithm names.
    metric_name : str, default='Score'
        Name of the metric.
    figsize : tuple, default=(12, 6)
        Figure size.
    save_path : str, optional
        Path to save the figure.
    show_mean : bool, default=True
        Show mean as a marker.
    notch : bool, default=False
        Show notched boxplot (confidence interval).
    show_grid : bool, default=False
        Whether to show axis grid lines.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    # Apply base style, then use a classic boxplot look.
    apply_style()

    if isinstance(scores, dict):
        names = list(scores.keys())
        scores = np.array([scores[name] for name in names]).T

    scores = np.asarray(scores)
    n_datasets, n_algorithms = scores.shape

    if names is None:
        names = [f'Alg{i+1}' for i in range(n_algorithms)]

    fig, ax = setup_figure(figsize=figsize)

    classic_colors = ['#FF0000', '#FF6F00', '#F3F300', '#66FF00', '#00C2FF', '#8A2BE2']
    box_colors = [classic_colors[i % len(classic_colors)] for i in range(n_algorithms)]

    bp = ax.boxplot(
        [scores[:, i] for i in range(n_algorithms)],
        labels=names,
        patch_artist=True,
        notch=notch,
        widths=0.8,
        medianprops={'color': '#000000', 'linewidth': 3.0},
        whiskerprops={'color': '#000000', 'linewidth': 1.2, 'linestyle': (0, (5, 4))},
        capprops={'color': '#000000', 'linewidth': 1.2},
        boxprops={'edgecolor': '#000000', 'linewidth': 1.6},
        flierprops={
            'marker': 'o',
            'markersize': 5,
            'markerfacecolor': '#000000',
            'markeredgecolor': '#000000',
            'alpha': 0.55,
        },
    )

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.95)

    if show_mean:
        means = [np.mean(scores[:, i]) for i in range(n_algorithms)]
        ax.scatter(
            range(1, n_algorithms + 1),
            means,
            marker='D',
            color='#B91C1C',
            s=70,
            zorder=4,
            edgecolors='white',
            linewidths=1.5,
            label='Mean',
        )
        ax.legend(loc='lower left', framealpha=0.95, edgecolor='#888888')

    style_axis(
        ax,
        title=f'{metric_name} Distribution by Algorithm',
        xlabel='Algorithm',
        ylabel=metric_name,
        legend=show_mean,
        grid=show_grid,
    )

    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)

    plt.show()

def plot_heatmap(
    scores: Union[np.ndarray, Dict[str, np.ndarray]],
    names: List[str] = None,
    dataset_names: List[str] = None,
    metric_name: str = 'Score',
    figsize: Tuple[int, int] = None,
    save_path: str = None,
    cmap: str = 'YlGnBu',
    annotate: bool = True,
    precision: int = 3,
    show_grid: bool = False,
):
    """
    Plot heatmap of scores.

    Parameters
    ----------
    scores : ndarray of shape (n_datasets, n_algorithms) or dict
        Performance scores matrix.
    names : list of str, optional
        Algorithm names.
    dataset_names : list of str, optional
        Dataset names.
    metric_name : str, default='Score'
        Name of the metric.
    figsize : tuple, optional
        Figure size.
    save_path : str, optional
        Path to save the figure.
    cmap : str, default='YlGnBu'
        Colormap name.
    annotate : bool, default=True
        Show values in cells.
    precision : int, default=3
        Decimal precision for annotations.
    show_grid : bool, default=False
        Whether to show axis grid lines.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    # Apply modern styling
    apply_style()

    if isinstance(scores, dict):
        names = list(scores.keys())
        scores = np.array([scores[name] for name in names]).T

    scores = np.asarray(scores)
    n_datasets, n_algorithms = scores.shape

    if names is None:
        names = [f'Alg{i+1}' for i in range(n_algorithms)]

    if dataset_names is None:
        dataset_names = [f'D{i+1}' for i in range(n_datasets)]

    if figsize is None:
        figsize = (max(10, n_algorithms * 1.2), max(6, n_datasets * 0.6))

    fig, ax = setup_figure(figsize=figsize)

    if HAS_SEABORN:
        fmt = f'.{precision}f'
        sns.heatmap(
            scores, annot=annotate, fmt=fmt, cmap=cmap,
            xticklabels=names, yticklabels=dataset_names,
            ax=ax, linewidths=1.0, linecolor='white',
            annot_kws={'fontsize': 13, 'fontweight': 'bold'},
        )
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.set_label(metric_name.title(), fontsize=13, fontweight='bold')
            cbar.ax.tick_params(labelsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    else:
        im = ax.imshow(scores, cmap=cmap, aspect='auto')

        # Labels
        ax.set_xticks(range(n_algorithms))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=12)
        ax.set_yticks(range(n_datasets))
        ax.set_yticklabels(dataset_names, fontsize=12)

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Annotate
        if annotate:
            for i in range(n_datasets):
                for j in range(n_algorithms):
                    text = f'{scores[i, j]:.{precision}f}'
                    # Choose text color based on background
                    normalized_val = (scores[i, j] - scores.min()) / (scores.max() - scores.min() + 1e-10)
                    text_color = 'white' if normalized_val > 0.6 or normalized_val < 0.4 else '#333333'
                    ax.text(j, i, text, ha='center', va='center',
                            color=text_color, fontsize=13, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(metric_name.title(), fontsize=13, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)

    ax.set_title(f'{metric_name} Heatmap'.title(), fontsize=16, fontweight='bold', pad=14)
    ax.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=14, fontweight='bold')
    ax.grid(show_grid)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)

    plt.show()
