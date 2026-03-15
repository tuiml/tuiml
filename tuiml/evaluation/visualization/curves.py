"""
Performance curve visualizations (ROC, PR, Learning curves).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# NumPy 2.x compatibility: trapz was renamed to trapezoid
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
if _trapz is None:
    def _trapz(y, x):
        return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ._style import get_colors, setup_figure, style_axis, SEMANTIC_COLORS

def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = 'ROC Curve',
    figsize: Tuple[int, int] = (8, 6),
    save_path: str = None,
    show_auc: bool = True,
    label: str = None,
    show_grid: bool = False,
):
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : ndarray
        True binary labels.
    y_score : ndarray
        Predicted probabilities for positive class.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show_auc : bool, default=True
        Show AUC score in legend.
    label : str, optional
        Label for the curve.
    show_grid : bool, default=False
        Whether to show axis grid lines.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    colors = get_colors(2)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Calculate ROC curve
    thresholds = np.unique(y_score)
    thresholds = np.sort(thresholds)[::-1]

    tpr_list = [0.0]
    fpr_list = [0.0]

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    tpr_list.append(1.0)
    fpr_list.append(1.0)

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)

    # Calculate AUC using trapezoidal rule
    sorted_idx = np.argsort(fpr)
    fpr_sorted = fpr[sorted_idx]
    tpr_sorted = tpr[sorted_idx]
    auc = _trapz(tpr_sorted, fpr_sorted)

    # Plot
    fig, ax = setup_figure(figsize=figsize)

    if label is None:
        label = f'ROC (AUC = {auc:.3f})' if show_auc else 'ROC'
    elif show_auc:
        label = f'{label} (AUC = {auc:.3f})'

    ax.plot(fpr_sorted, tpr_sorted, lw=3.0, label=label, color=colors[0])
    ax.plot([0, 1], [0, 1], '--', lw=2.0, label='Random', color=SEMANTIC_COLORS['neutral'])

    # Fill area under curve
    ax.fill_between(fpr_sorted, tpr_sorted, alpha=0.2, color=colors[0])

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    style_axis(
        ax,
        title=title,
        xlabel='False Positive Rate',
        ylabel='True Positive Rate',
        legend=True,
        legend_loc='lower right',
        grid=True,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)

    plt.show()

    return fpr_sorted, tpr_sorted, auc

def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = 'Precision-Recall Curve',
    figsize: Tuple[int, int] = (8, 6),
    save_path: str = None,
    show_ap: bool = True,
    label: str = None,
    show_grid: bool = False,
):
    """
    Plot Precision-Recall curve.

    Parameters
    ----------
    y_true : ndarray
        True binary labels.
    y_score : ndarray
        Predicted probabilities for positive class.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    show_ap : bool, default=True
        Show Average Precision in legend.
    label : str, optional
        Label for the curve.
    show_grid : bool, default=False
        Whether to show axis grid lines.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    colors = get_colors(2)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Calculate PR curve
    thresholds = np.unique(y_score)
    thresholds = np.sort(thresholds)[::-1]

    precision_list = [1.0]
    recall_list = [0.0]

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    precision = np.array(precision_list)
    recall = np.array(recall_list)

    # Calculate Average Precision
    sorted_idx = np.argsort(recall)
    recall_sorted = recall[sorted_idx]
    precision_sorted = precision[sorted_idx]
    ap = _trapz(precision_sorted, recall_sorted)

    # Plot
    fig, ax = setup_figure(figsize=figsize)

    if label is None:
        label = f'PR (AP = {ap:.3f})' if show_ap else 'PR'
    elif show_ap:
        label = f'{label} (AP = {ap:.3f})'

    ax.plot(recall_sorted, precision_sorted, lw=3.0, label=label, color=colors[0])

    # Fill area under curve
    ax.fill_between(recall_sorted, precision_sorted, alpha=0.2, color=colors[0])

    # Baseline
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color=SEMANTIC_COLORS['neutral'], linestyle='--', lw=2.0,
               label=f'Baseline ({baseline:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    style_axis(
        ax,
        title=title,
        xlabel='Recall',
        ylabel='Precision',
        legend=True,
        legend_loc='lower left',
        grid=True,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)

    plt.show()

    return recall_sorted, precision_sorted, ap

def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    title: str = 'Learning Curve',
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None,
    metric_name: str = 'Score',
    show_std: bool = True,
    show_grid: bool = False,
):
    """
    Plot learning curve showing performance vs training set size.

    Parameters
    ----------
    train_sizes : ndarray
        Training set sizes.
    train_scores : ndarray of shape (n_sizes,) or (n_sizes, n_splits)
        Training scores.
    test_scores : ndarray of shape (n_sizes,) or (n_sizes, n_splits)
        Test/validation scores.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.
    metric_name : str
        Name of the metric.
    show_std : bool, default=True
        Show standard deviation bands.
    show_grid : bool, default=False
        Whether to show axis grid lines.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    colors = get_colors(2)

    train_sizes = np.asarray(train_sizes)
    train_scores = np.asarray(train_scores)
    test_scores = np.asarray(test_scores)

    # Handle both 1D and 2D arrays
    if train_scores.ndim == 1:
        train_mean = train_scores
        train_std = np.zeros_like(train_mean)
        test_mean = test_scores
        test_std = np.zeros_like(test_mean)
    else:
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

    fig, ax = setup_figure(figsize=figsize)

    # Training curve
    ax.plot(train_sizes, train_mean, 'o-', color=colors[0], lw=3.0,
            markersize=10, label='Training score')
    if show_std and train_scores.ndim > 1:
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color=colors[0])

    # Validation curve
    ax.plot(train_sizes, test_mean, 's-', color=colors[1], lw=3.0,
            markersize=10, label='Cross-validation score')
    if show_std and test_scores.ndim > 1:
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                        alpha=0.2, color=colors[1])

    style_axis(
        ax,
        title=title,
        xlabel='Training Set Size',
        ylabel=metric_name,
        legend=True,
        legend_loc='best',
        grid=True,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)

    plt.show()
