"""
Modern visualization styling for TuiML.

Provides consistent, publication-quality plot styling across all visualizations.
"""

import numpy as np
from typing import Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Modern color palettes
PALETTES = {
    'default': [
        '#4C72B0',  # Steel blue
        '#DD8452',  # Coral
        '#55A868',  # Sage green
        '#C44E52',  # Brick red
        '#8172B3',  # Lavender
        '#937860',  # Taupe
        '#DA8BC3',  # Pink
        '#8C8C8C',  # Gray
        '#CCB974',  # Olive
        '#64B5CD',  # Sky blue
    ],
    'vibrant': [
        '#0077B6',  # Deep blue
        '#E63946',  # Red
        '#2A9D8F',  # Teal
        '#E9C46A',  # Yellow
        '#9B2335',  # Burgundy
        '#264653',  # Dark teal
        '#F4A261',  # Orange
        '#A855F7',  # Purple
    ],
    'muted': [
        '#6B7280',  # Gray
        '#3B82F6',  # Blue
        '#10B981',  # Emerald
        '#F59E0B',  # Amber
        '#EF4444',  # Red
        '#8B5CF6',  # Violet
        '#EC4899',  # Pink
        '#14B8A6',  # Teal
    ],
    'scientific': [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
    ],
}

# Semantic color tokens for consistent, professional visuals
SEMANTIC_COLORS = {
    'primary': '#2563EB',
    'secondary': '#0F766E',
    'success': '#16A34A',
    'warning': '#D97706',
    'danger': '#DC2626',
    'neutral': '#6B7280',
    'text': '#1F2937',
    'muted_text': '#6B7280',
    'grid': '#E5E7EB',
    'border': '#D1D5DB',
}

# Default style configuration
STYLE_CONFIG = {
    # Figure
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'figure.dpi': 100,
    'figure.autolayout': True,

    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans', 'sans-serif'],
    'font.size': 12,
    'font.weight': 'medium',

    # Axes
    'axes.facecolor': 'white',
    'axes.edgecolor': '#111111',
    'axes.linewidth': 1.5,
    'axes.grid': False,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.titlepad': 14,
    'axes.labelsize': 14,
    'axes.labelweight': 'bold',
    'axes.labelpad': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelcolor': SEMANTIC_COLORS['text'],
    'axes.titlecolor': SEMANTIC_COLORS['text'],
    'axes.prop_cycle': None,  # Set dynamically

    # Grid
    'grid.color': SEMANTIC_COLORS['grid'],
    'grid.linewidth': 0.8,
    'grid.alpha': 0.8,
    'grid.linestyle': '-',

    # Ticks
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.color': SEMANTIC_COLORS['text'],
    'ytick.color': SEMANTIC_COLORS['text'],
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,

    # Legend
    'legend.fontsize': 12,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.facecolor': 'white',
    'legend.edgecolor': '#999999',
    'legend.borderpad': 0.5,
    'legend.labelspacing': 0.4,

    # Lines
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'lines.markeredgewidth': 0,

    # Patches (bars, etc.)
    'patch.linewidth': 1.0,
    'patch.edgecolor': 'white',

    # Savefig
    'savefig.dpi': 300,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}


def apply_style(palette: str = 'default', dark_mode: bool = False) -> None:
    """
    Apply modern styling to matplotlib plots.

    Parameters
    ----------
    palette : str, default='default'
        Color palette name. Options: 'default', 'vibrant', 'muted', 'scientific'.
    dark_mode : bool, default=False
        Enable dark mode styling.

    Examples
    --------
    >>> from tuiml.evaluation.visualization._style import apply_style
    >>> apply_style()  # Apply default modern style
    >>> apply_style(palette='vibrant')  # Use vibrant colors
    """
    if not HAS_MATPLOTLIB:
        return

    # Get color palette
    colors = PALETTES.get(palette, PALETTES['default'])

    # Apply base configuration
    for key, value in STYLE_CONFIG.items():
        if value is not None and key != 'axes.prop_cycle':
            try:
                mpl.rcParams[key] = value
            except (KeyError, ValueError):
                pass

    # Set color cycle
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

    # Dark mode adjustments
    if dark_mode:
        dark_config = {
            'figure.facecolor': '#1F2937',
            'figure.edgecolor': '#1F2937',
            'axes.facecolor': '#1F2937',
            'axes.edgecolor': '#9CA3AF',
            'axes.labelcolor': '#F3F4F6',
            'text.color': '#F3F4F6',
            'xtick.color': '#9CA3AF',
            'ytick.color': '#9CA3AF',
            'grid.color': '#374151',
            'legend.facecolor': '#374151',
            'legend.edgecolor': '#4B5563',
            'savefig.facecolor': '#1F2937',
            'savefig.edgecolor': '#1F2937',
        }
        for key, value in dark_config.items():
            try:
                mpl.rcParams[key] = value
            except (KeyError, ValueError):
                pass


def reset_style() -> None:
    """Reset matplotlib to default styling."""
    if HAS_MATPLOTLIB:
        mpl.rcdefaults()


def get_colors(n: int = None, palette: str = 'default') -> list:
    """
    Get colors from the specified palette.

    Parameters
    ----------
    n : int, optional
        Number of colors to return. If None, returns all colors.
    palette : str, default='default'
        Palette name.

    Returns
    -------
    colors : list
        List of color hex codes.
    """
    colors = PALETTES.get(palette, PALETTES['default'])
    if n is None:
        return colors
    if n <= len(colors):
        return colors[:n]
    # Cycle colors if more are needed
    return [colors[i % len(colors)] for i in range(n)]


def setup_figure(
    figsize: tuple = (10, 6),
    palette: str = 'default',
    dark_mode: bool = False,
    style: str = None
) -> tuple:
    """
    Create a styled figure and axes.

    Parameters
    ----------
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    palette : str, default='default'
        Color palette name.
    dark_mode : bool, default=False
        Enable dark mode.
    style : str, optional
        Additional matplotlib style to apply (e.g., 'seaborn-v0_8-whitegrid').

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.

    Examples
    --------
    >>> from tuiml.evaluation.visualization._style import setup_figure
    >>> fig, ax = setup_figure(figsize=(12, 8))
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> plt.show()
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")

    # Apply base style
    apply_style(palette=palette, dark_mode=dark_mode)

    # Apply additional style if specified
    if style:
        try:
            plt.style.use(style)
        except (OSError, ValueError):
            pass

    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def style_axis(
    ax,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    legend: bool = True,
    legend_loc: str = 'best',
    grid: bool = False,
    despine: bool = True,
) -> None:
    """
    Apply consistent styling to an axis.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    title : str, optional
        Axis title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    legend : bool, default=True
        Show legend if handles exist.
    legend_loc : str, default='best'
        Legend location.
    grid : bool, default=False
        Show grid.
    despine : bool, default=True
        Remove top and right spines.
    """
    if title:
        ax.set_title(title.title(), fontsize=16, fontweight='bold', pad=14)
    if xlabel:
        ax.set_xlabel(xlabel.title(), fontsize=14, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel.title(), fontsize=14, fontweight='bold')

    if grid:
        ax.grid(
            True,
            alpha=STYLE_CONFIG['grid.alpha'],
            linewidth=STYLE_CONFIG['grid.linewidth'],
            color=STYLE_CONFIG['grid.color'],
        )
    else:
        ax.grid(False)

    if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc=legend_loc, framealpha=0.95, edgecolor='#999999')


def annotate_bars(
    ax,
    bars,
    fmt: str = '.2f',
    offset: float = 3,
    fontsize: int = 11,
    color: str = '#111111',
) -> None:
    """
    Add value annotations to bar chart.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes.
    bars : BarContainer
        Bar container from ax.bar().
    fmt : str, default='.2f'
        Number format string.
    offset : float, default=3
        Vertical offset for text.
    fontsize : int, default=9
        Font size for annotations.
    color : str, default='#333333'
        Text color.
    """
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:{fmt}}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=fontsize,
            color=color,
        )


# Auto-apply style when module is imported
if HAS_MATPLOTLIB:
    apply_style()
